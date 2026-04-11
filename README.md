# medbench-eval

**LLM-as-judge evaluation for medical AI — built on the MedBench-Agent-95 benchmark.**

Evaluate any LLM's clinical capabilities across 13 agentic medical tasks using structured rubrics and automated scoring. Works as a standalone Python tool, a Claude Code skill (`/medbench-eval`), and a CI/CD pipeline component.

---

## What is this?

`medbench-eval` is a framework for evaluating medical AI models using the **LLM-as-judge** methodology, calibrated against the **MedBench-Agent-95** gold-standard benchmark (390 samples across 13 clinical task categories).

**Key design decisions:**

- **Substance over style**: the judge ignores formatting, phrasing, and step ordering — only clinical correctness matters
- **Task-specific rubrics**: each of the 13 tasks has 4-5 Likert criteria designed for that task's clinical requirements
- **Capability-first UX**: group tasks by what clinical capability you're testing, not by task name
- **DUT tracking**: every evaluation report identifies the model under test alongside the judge model
- **Model-agnostic judge**: supports `claude-haiku-4-5` (default) and `minimax-m2.5`

---

## Benchmark Tasks

| Capability | Tasks | What it evaluates |
|------------|-------|-------------------|
| Clinical Reasoning | MedCOT, MedDecomp, MedPathPlan | Differential diagnosis, task decomposition, treatment pathway planning |
| Long-context Understanding | MedLongQA, MedLongConv | Long document Q&A, multi-turn conversation memory |
| Agentic Tool Use | MedCallAPI, MedRetAPI, MedDBOps | API calls, knowledge retrieval, database queries |
| Multi-system Orchestration | MedCollab | Coordinating multiple systems for complex medical goals |
| Self-reflection | MedReflect | Identifying and correcting clinical errors |
| Role Adaptation | MedRoleAdapt | Adapting communication to patient/nurse/physician roles |
| Safety & Defense | MedShield, MedDefend | Refusing harmful requests, resisting adversarial inputs |

Each task has a gold-standard answer used to calibrate what "correct" looks like. The judge scores 1-5 per criterion; scores are normalized to 0-100.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/medqa-auto-eval-skills
cd medqa-auto-eval-skills

# 2. Install
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the interactive wizard
python eval.py
```

The wizard guides you through:
1. Name the model you're evaluating (DUT)
2. Select which clinical capability to test
3. Choose sample count (Quick=3 / Standard=5 / Thorough=10 / Full=30)
4. Confirm paths and start

---

## Installation

**Requirements:** Python 3.11+

```bash
pip install -r requirements.txt
```

**Judge model options:**

| Model | Env var | Notes |
|-------|---------|-------|
| `claude-haiku-4-5` (default) | `ANTHROPIC_API_KEY` | Fast, cost-effective |
| `minimax-m2.5` | `MINIMAX_API_KEY` | Alternative judge model |

---

## Usage

### Interactive wizard (recommended)

```bash
python eval.py
```

Prompts for DUT name, capability group, sample count, and paths. No flags needed.

### Batch evaluation by capability

```bash
# Evaluate clinical reasoning of a model
python eval.py batch \
  --benchmark medbench-agent-95/ \
  --capability reasoning \
  --dut gpt-4o \
  --responses-dir outputs/gpt-4o/ \
  --samples 5 \
  --output results/gpt-4o-reasoning/

# Evaluate safety & defense
python eval.py batch \
  --benchmark medbench-agent-95/ \
  --capability safety \
  --dut my-model-v2 \
  --responses-dir outputs/my-model/ \
  --output results/safety/

# Full benchmark (all 13 tasks)
python eval.py batch \
  --benchmark medbench-agent-95/ \
  --capability full \
  --dut claude-opus-4-6 \
  --responses-dir outputs/opus/ \
  --samples 10 \
  --output results/opus-full/
```

**Capability group keys:** `reasoning`, `long_context`, `tool_use`, `orchestration`, `self_correction`, `role_adapt`, `safety`, `full`

### Single response evaluation

```bash
python eval.py single \
  --task MedCOT \
  --dut my-model \
  --question "Patient, 45M, fever 39°C, neck stiffness 3 days..." \
  --response "Based on the presentation, I suspect bacterial meningitis..." \
  --gold-answer "Step 1: Key features include fever, meningismus..."
```

Output is JSON:
```json
{
  "task": "MedCOT",
  "dut": "my-model",
  "judge_model": "claude-haiku-4-5",
  "total_score": 4.2,
  "normalized_score": 80.0,
  "criteria": [...],
  "overall_feedback": "Strong reasoning chain...",
  "minor_errors": ["Used 'meningococcal' instead of 'Neisseria meningitidis'"],
  "major_errors": []
}
```

### Calibration baseline (gold answers)

Evaluate the benchmark's gold answers to establish a score ceiling:

```bash
python eval.py batch \
  --benchmark medbench-agent-95/ \
  --capability full \
  --samples 5 \
  --output results/gold-ceiling/
```

Gold answers typically score 85-95/100 — use this as your upper bound.

### List tasks and capabilities

```bash
python eval.py tasks
```

---

## Scoring

| Component | Range | Description |
|-----------|-------|-------------|
| Criterion score | 1-5 | Per-criterion Likert score (4-5 criteria per task) |
| Total score | 1.0-5.0 | Average of criterion scores |
| Normalized score | 0-100 | `(total - 1) / 4 × 100` |

**Normalized score interpretation:**

| Score | Interpretation |
|-------|----------------|
| 90-100 | Exceptional — matches or exceeds gold standard |
| 75-89 | Strong — minor gaps only |
| 60-74 | Adequate — some important gaps |
| 40-59 | Weak — significant deficiencies |
| 0-39 | Poor — major clinical failures |

### Robustness policy

The judge explicitly separates:
- **Minor errors** (no score impact): different phrasing, formatting, step ordering, synonym terminology, extra correct detail
- **Major errors** (score impact): wrong diagnosis, wrong drug/dose, missing safety-critical steps, incorrect lab interpretation, logical inconsistency

Both lists appear in every evaluation result, giving you full transparency.

---

## Output Files

After a batch run, `--output results/` contains:

| File | Description |
|------|-------------|
| `summary.json` | Structured summary with per-task stats, DUT, judge model, error counts |
| `details.jsonl` | One record per sample: scores, criteria, minor/major errors, token usage |
| `report.md` | Human-readable markdown report with tables and per-criterion breakdown |

Example `summary.json`:
```json
{
  "meta": {
    "timestamp": "2026-04-11T10:00:00Z",
    "judge_model": "claude-haiku-4-5",
    "dut": "gpt-4o",
    "tasks_evaluated": ["MedCOT", "MedDecomp", "MedPathPlan"],
    "total_samples": 15,
    "total_minor_errors": 8,
    "total_major_errors": 2
  },
  "overall": { "avg_score": 76.4 },
  "tasks": {
    "MedCOT": { "n": 5, "avg_score": 78.0, "criterion_means": {...} }
  }
}
```

---

## Providing Model Responses

The `--responses-dir` should contain one `.txt` file per sample, organized by task and sample ID:

```
outputs/
  my-model/
    MedCOT/
      97.txt        ← response for sample ID 97
      103.txt
    MedDecomp/
      12.txt
```

Sample IDs come from the `other.id` field in each JSONL record. If a response file is missing, that sample is skipped and logged as an error.

If `--responses-dir` is omitted, the benchmark's own gold answers are evaluated (useful for calibration).

---

## Claude Code Skill

Add this repo to Claude Code and use `/medbench-eval` directly in your session:

```
/medbench-eval
```

The skill (defined in `SKILL.md`) guides you through the same wizard interactively — no terminal needed. Claude acts as the judge inline for single-response evaluation, or delegates to the Python CLI for batch runs.

**To install:**
1. Add this repo's `SKILL.md` to your Claude Code skills directory, or
2. Reference it via the Claude Code skills marketplace

---

## Repository Structure

```
medqa-auto-eval-skills/
├── README.md
├── SKILL.md                    ← Claude Code /medbench-eval skill
├── eval.py                     ← CLI entry point (interactive wizard + subcommands)
├── requirements.txt
├── judge/
│   ├── capabilities.py         ← 8 capability groups → task mapping
│   ├── rubrics.py              ← 13 task rubrics (57 criteria total)
│   ├── judge.py                ← LLM-as-judge core + robustness policy
│   ├── models.py               ← claude-haiku-4-5 / minimax-m2.5 abstraction
│   ├── runner.py               ← batch evaluation loop
│   └── report.py               ← JSON + markdown report generation
└── medbench-agent-95/          ← benchmark data (390 samples, 13 tasks)
    ├── MedCOT.jsonl + .md
    ├── MedCallAPI.jsonl + .md
    └── ... (13 tasks × 30 samples each)
```

---

## Task Rubrics

Each task is evaluated on 4-5 criteria. Summary:

| Task | Criteria |
|------|----------|
| MedCOT | step_completeness, clinical_accuracy, differential_diagnosis_quality, evidence_integration, conclusion_clarity |
| MedCallAPI | api_selection, parameter_completeness, format_correctness, error_handling |
| MedCollab | task_decomposition, system_coordination, workflow_logic, completeness |
| MedDBOps | query_correctness, clinical_alignment, efficiency, data_integrity |
| MedDecomp | completeness, logical_ordering, granularity, clinical_feasibility |
| MedDefend | adversarial_detection, refusal_quality, safety_rationale, safe_alternative |
| MedLongConv | memory_accuracy, consistency, clinical_continuity, response_relevance |
| MedLongQA | answer_accuracy, completeness, source_grounding, clarity |
| MedPathPlan | guideline_adherence, individualization, completeness, temporal_logic |
| MedReflect | error_identification, correction_quality, reasoning_depth, improvement_actionability |
| MedRetAPI | query_precision, query_completeness, retrieval_strategy, result_relevance |
| MedRoleAdapt | role_fidelity, communication_style, clinical_accuracy, empathy_professionalism |
| MedShield | risk_recognition, intervention_timeliness, intervention_appropriateness, harm_prevention_effectiveness |

Full criterion definitions (with score-1 and score-5 anchors) are in `judge/rubrics.py`.

---

## Benchmark Data

The `medbench-agent-95/` directory contains 390 samples from the [MedBench](https://medbench.opencompass.org.cn/) evaluation suite — 30 samples per task, each with:

- `question`: clinical scenario or task prompt
- `answer`: gold standard response
- `other`: metadata (`id`, `source`)

The benchmark covers agentic medical tasks (not simple MCQ) and uses LLM-as-judge as the primary evaluation methodology — making this repo the natural evaluation harness for it.

---

## Contributing

Contributions welcome:
- **New task rubrics** — add to `judge/rubrics.py` and `judge/capabilities.py`
- **New judge models** — add a backend to `judge/models.py`
- **Benchmark extensions** — add JSONL files to `medbench-agent-95/`
- **Bug reports** — open an issue

---

## License

MIT
