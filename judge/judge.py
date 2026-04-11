"""
Core LLM-as-judge logic for MedBench-Agent-95 evaluation.
Scores model responses against task-specific rubrics using a 5-point Likert scale.

Robustness policy: minor surface errors (formatting, phrasing, step ordering) do NOT
lower scores. Only substantive clinical errors affect scoring.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .rubrics import Rubric, get_rubric
from .models import call_model, ModelResponse, DEFAULT_MODEL


@dataclass
class CriterionScore:
    name: str
    score: int          # 1-5
    justification: str
    max_score: int = 5


@dataclass
class JudgementResult:
    task: str
    criterion_scores: list[CriterionScore]
    total_score: float          # average of criterion scores (1.0-5.0)
    normalized_score: float     # (total_score - 1) / 4 * 100, range 0-100
    overall_feedback: str
    model: str                  # judge model
    dut: str                    # device/model under test
    input_tokens: int
    output_tokens: int
    minor_errors: list[str] = field(default_factory=list)   # surface issues, no score impact
    major_errors: list[str] = field(default_factory=list)   # clinical errors, score impact
    error: Optional[str] = None


SYSTEM_PROMPT = """You are an expert medical AI evaluator. Your role is to assess medical AI responses using structured rubrics.

You must return ONLY valid JSON — no markdown, no explanation outside the JSON structure.

## Scoring Philosophy

SUBSTANCE over STYLE. Apply these rules strictly:

**DO NOT penalize for:**
- Different phrasing or wording when the clinical meaning is equivalent
- Markdown formatting differences (using bullets vs. numbered lists vs. prose)
- Reordering of steps when order doesn't affect clinical correctness
- Minor spelling or grammatical errors that don't obscure meaning
- Providing MORE detail than the gold answer (extra correct information is a bonus, not a flaw)
- Different but equally valid clinical terminology for the same concept

**DO penalize for:**
- Wrong diagnosis, wrong drug, wrong dose, wrong procedure
- Missing a step that is clinically critical for patient safety
- Incorrect interpretation of lab values or clinical findings
- Logically inconsistent reasoning that would mislead a clinician
- Refusing to answer when an answer is required
- Fabricating clinical information not supported by the question

A score of 3 is average. Reserve 5 for responses that fully satisfy all criteria (surface errors allowed).
Reserve 1 only for responses that would cause clinical harm or are completely off-task."""


def _build_judge_prompt(
    task: str,
    rubric: Rubric,
    question: str,
    response: str,
    gold_answer: Optional[str],
    dut: str = "unknown",
    anchor_examples: Optional[list[dict]] = None,
) -> str:
    """
    Build the judge prompt.

    Args:
        anchor_examples: Optional list of calibration examples, each a dict with
                         'question' and 'answer' keys drawn from gold benchmark.
                         When provided, they are shown BEFORE the current question
                         to anchor the judge's score-5 interpretation.
    """
    criteria_text = "\n".join(
        f"""  - **{c.name}**: {c.description}
    Score 1: {c.score_1}
    Score 5: {c.score_5}"""
        for c in rubric.criteria
    )

    gold_section = ""
    if gold_answer:
        gold_section = f"""
## Gold Standard Answer (Reference for THIS question)
Use this ONLY to calibrate what "correct" looks like for this specific question.
Do NOT require the model response to match the gold answer word-for-word or structure-for-structure.
Award full credit when the model achieves the same clinical correctness through different means.
```
{gold_answer[:2000]}{"..." if len(gold_answer) > 2000 else ""}
```
"""

    # Few-shot calibration block: concrete examples of score-5 answers for this task
    anchor_section = ""
    if anchor_examples:
        parts = [
            "## Score-5 Calibration Examples",
            f"The following are GOLD STANDARD answers for {task} tasks that deserve a score of 5 on all criteria.",
            "Study these to calibrate your understanding of what an ideal response looks like for this task type.",
            "Do NOT penalize the evaluated response for not matching these examples verbatim.",
            "",
        ]
        for idx, ex in enumerate(anchor_examples, 1):
            q_preview = str(ex.get("question", ""))[:500]
            a_preview = str(ex.get("answer", ""))[:1000]
            parts += [
                f"### Calibration Example {idx}",
                f"**Question (excerpt):** {q_preview}{'...' if len(str(ex.get('question',''))) > 500 else ''}",
                f"**Ideal Answer (score 5):**",
                f"```",
                f"{a_preview}{'...' if len(str(ex.get('answer',''))) > 1000 else ''}",
                f"```",
                "",
            ]
        anchor_section = "\n".join(parts)

    criteria_names = [c.name for c in rubric.criteria]
    example_scores = {name: {"score": 4, "justification": "..."} for name in criteria_names}
    example_json = {
        "criterion_scores": example_scores,
        "overall_feedback": "Brief overall assessment...",
        "minor_errors_noted": ["list any surface-level issues that did NOT affect the score"],
        "major_errors_noted": ["list any substantive clinical errors that DID affect the score"],
    }

    anchor_block = f"\n{anchor_section}\n" if anchor_section else ""

    return f"""# Medical AI Response Evaluation

## Task: {task}
## DUT (Model Under Test): {dut}
{rubric.description}
{anchor_block}
## Question / Clinical Scenario
```
{question[:3000]}{"..." if len(question) > 3000 else ""}
```
{gold_section}
## Model Response to Evaluate
```
{response[:3000]}{"..." if len(response) > 3000 else ""}
```

## Evaluation Rubric
Score each criterion from 1 (worst) to 5 (best):

{criteria_text}

## Robustness Rules (MANDATORY)
- Surface errors (phrasing, formatting, step ordering) do NOT lower scores
- Only substantive clinical errors that affect patient safety or correctness lower scores
- A response with minor phrasing differences but correct clinical content scores 4-5
- Extra correct information beyond the gold standard is a strength, not a flaw

## Instructions
1. Evaluate SUBSTANCE, not surface style
2. Assign a score of 1-5 for each criterion with a 1-2 sentence justification
3. Separately list minor errors (no score impact) and major errors (score impact)
4. Give an overall 2-3 sentence summary

Return ONLY this JSON structure (no markdown, no preamble):
{json.dumps(example_json, indent=2, ensure_ascii=False)}"""


def _parse_judge_response(content: str, rubric: Rubric) -> tuple[list[CriterionScore], str, list[str], list[str]]:
    """Parse the judge's JSON response into structured scores."""
    # Strip markdown code blocks if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    data = json.loads(cleaned)

    scores = []
    criterion_scores_raw = data.get("criterion_scores", {})

    for criterion in rubric.criteria:
        raw = criterion_scores_raw.get(criterion.name, {})
        if isinstance(raw, dict):
            score_val = int(raw.get("score", 3))
            justification = str(raw.get("justification", ""))
        elif isinstance(raw, (int, float)):
            score_val = int(raw)
            justification = ""
        else:
            score_val = 3
            justification = ""

        score_val = max(1, min(5, score_val))
        scores.append(CriterionScore(
            name=criterion.name,
            score=score_val,
            justification=justification,
        ))

    overall_feedback = str(data.get("overall_feedback", ""))
    minor_errors = [str(e) for e in data.get("minor_errors_noted", [])]
    major_errors = [str(e) for e in data.get("major_errors_noted", [])]
    return scores, overall_feedback, minor_errors, major_errors


def judge_response(
    task: str,
    question: str,
    response: str,
    gold_answer: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    dut: str = "unknown",
    anchor_examples: Optional[list[dict]] = None,
) -> JudgementResult:
    """
    Evaluate a model response against the task rubric using LLM-as-judge.

    Args:
        task: Task name (e.g. "MedCOT", "MedCallAPI")
        question: The clinical question/scenario
        response: The model response to evaluate
        gold_answer: Optional gold standard answer for calibration
        model: Judge model identifier
        dut: Name of the model/system being evaluated (Device Under Test)
        anchor_examples: Optional list of {question, answer} dicts from the gold benchmark.
                         When provided, shown to the judge as concrete score-5 calibration
                         examples BEFORE the actual question. Improves alignment on tasks
                         where the judge misunderstands the task type (low ceiling problem).

    Returns:
        JudgementResult with per-criterion scores, minor/major error lists, and overall feedback
    """
    rubric = get_rubric(task)
    prompt = _build_judge_prompt(
        task, rubric, question, response, gold_answer, dut=dut,
        anchor_examples=anchor_examples,
    )

    try:
        model_response: ModelResponse = call_model(prompt, model=model, system=SYSTEM_PROMPT)
        content = model_response.content
    except Exception as e:
        return JudgementResult(
            task=task,
            criterion_scores=[],
            total_score=0.0,
            normalized_score=0.0,
            overall_feedback="",
            model=model,
            dut=dut,
            input_tokens=0,
            output_tokens=0,
            error=str(e),
        )

    try:
        criterion_scores, overall_feedback, minor_errors, major_errors = _parse_judge_response(content, rubric)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return JudgementResult(
            task=task,
            criterion_scores=[],
            total_score=0.0,
            normalized_score=0.0,
            overall_feedback=content[:500],
            model=model,
            dut=dut,
            input_tokens=model_response.input_tokens,
            output_tokens=model_response.output_tokens,
            error=f"Parse error: {e}",
        )

    total = sum(s.score for s in criterion_scores) / len(criterion_scores) if criterion_scores else 0.0
    normalized = (total - 1) / 4 * 100 if criterion_scores else 0.0

    return JudgementResult(
        task=task,
        criterion_scores=criterion_scores,
        total_score=round(total, 3),
        normalized_score=round(normalized, 2),
        overall_feedback=overall_feedback,
        minor_errors=minor_errors,
        major_errors=major_errors,
        model=model,
        dut=dut,
        input_tokens=model_response.input_tokens,
        output_tokens=model_response.output_tokens,
    )


def judge_against_gold(
    task: str,
    question: str,
    gold_answer: str,
    model: str = DEFAULT_MODEL,
    dut: str = "gold-standard",
    anchor_examples: Optional[list[dict]] = None,
) -> JudgementResult:
    """
    Evaluate the gold standard answer itself to establish a ceiling score.
    Useful for calibrating what a perfect response looks like.
    """
    return judge_response(
        task=task,
        question=question,
        response=gold_answer,
        gold_answer=gold_answer,
        model=model,
        dut=dut,
        anchor_examples=anchor_examples,
    )
