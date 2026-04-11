"""
Judge-human alignment validation for MedBench-Agent-95.

MODE 1 — Synthetic tier test  (--task / --all-tasks)
  Tests discrimination between three known-quality tiers:
  TIER 1 — GOLD    : benchmark answer (human-curated, expect ≥ 85/100)
  TIER 2 — MISMATCH: another sample's answer (wrong for this Q, expect ≤ 55/100)
  TIER 3 — REFUSAL : "Cannot determine." (off-task, expect ≤ 20/100)

MODE 2 — Real model comparison  (--compare-dir)
  Compares gold answers vs real DUT responses for the SAME questions.
  Tests whether the judge reproduces a known human-established score gap.
  Example: gold (95) vs GPT-4.1 (81) → judge should show ~14-point gap.

Metrics reported (both modes):
  ceiling_score    — avg judge score on GOLD responses
  model_score      — avg judge score on DUT/MISMATCH responses [mode 2 / mode 1]
  gap              — ceiling − model_score  (target: ≥ expected external gap)
  discrimination   — % of samples where gold > model
  spearman_rho     — rank correlation between expected and actual score order

Usage:
  # Mode 1 — synthetic tiers
  python validate.py --benchmark medbench-agent-95/ --task MedCOT --samples 5
  python validate.py --benchmark medbench-agent-95/ --all-tasks --samples 3

  # Mode 2 — real model comparison
  python validate.py --benchmark medbench-agent-95/ --all-tasks --samples 5 \\
      --compare-dir results-gpt-4.1-20251216-score-81/MedBench_Agent \\
      --compare-name gpt-4.1 --compare-expected-score 81
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from judge.judge import judge_response
from judge.models import DEFAULT_MODEL
from judge.rubrics import list_tasks
from judge.runner import _load_anchor_examples
from judge.rag import GoldRAG


# Refusal string used for TIER-3 (minimal / off-task) test
_REFUSAL_TEXT = "无法判断。该问题超出了我的能力范围，请咨询专业医生。"


@dataclass
class TierResult:
    tier: str          # "gold" | "mismatch" | "refusal"
    sample_id: int
    question: str
    response: str
    gold_answer: str
    normalized_score: float
    error: Optional[str] = None


@dataclass
class SampleAlignment:
    sample_id: int
    task: str
    gold_score: float
    mismatch_score: float
    refusal_score: float
    gold_beats_bad: bool   # gold > max(mismatch, refusal)  — primary discrimination signal


@dataclass
class TaskAlignment:
    task: str
    n_samples: int
    ceiling_score: float         # avg gold score
    mismatch_score: float        # avg mismatch score
    refusal_score: float         # avg refusal score
    discrimination_rate: float   # % of triples correctly ordered
    spearman_rho: float          # rank correlation
    samples: list[SampleAlignment] = field(default_factory=list)


def _load_samples(benchmark_dir: Path, task: str) -> list[dict]:
    jsonl_path = benchmark_dir / f"{task}.jsonl"
    if not jsonl_path.exists():
        return []
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


def _spearman_rho(expected: list[float], actual: list[float]) -> float:
    """Compute Spearman rank correlation between two score lists (no external deps)."""
    n = len(expected)
    if n < 2:
        return 0.0

    def rank(lst: list[float]) -> list[float]:
        sorted_idx = sorted(range(n), key=lambda i: lst[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and lst[sorted_idx[j]] == lst[sorted_idx[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks

    r_exp = rank(expected)
    r_act = rank(actual)
    d2 = sum((r_exp[i] - r_act[i]) ** 2 for i in range(n))
    rho = 1 - 6 * d2 / (n * (n ** 2 - 1))
    return round(rho, 4)


def _judge_tier(
    task: str,
    question: str,
    response: str,
    gold_answer: str,
    tier: str,
    model: str,
    sample_id: int,
    delay: float,
) -> TierResult:
    try:
        result = judge_response(
            task=task,
            question=question,
            response=response,
            gold_answer=gold_answer,
            model=model,
            dut=f"validate-{tier}",
        )
        score = result.normalized_score
        error = result.error
    except Exception as e:
        score = 0.0
        error = str(e)

    time.sleep(delay)
    return TierResult(
        tier=tier,
        sample_id=sample_id,
        question=question,
        response=response,
        gold_answer=gold_answer,
        normalized_score=score,
        error=error,
    )


def validate_task(
    task: str,
    benchmark_dir: Path,
    n_samples: int,
    model: str,
    delay: float,
    verbose: bool,
) -> TaskAlignment:
    all_samples = _load_samples(benchmark_dir, task)
    if not all_samples:
        raise ValueError(f"No samples found for task: {task}")

    # Take first n_samples (deterministic, reproducible)
    samples = all_samples[:n_samples]
    n = len(samples)

    sample_alignments: list[SampleAlignment] = []
    gold_scores, mismatch_scores, refusal_scores = [], [], []

    for i, sample in enumerate(samples):
        question = str(sample.get("question", ""))
        gold_answer = str(sample.get("answer", ""))
        other = sample.get("other", {})
        sample_id = other.get("id", i) if isinstance(other, dict) else i

        # TIER 2: offset sample answer (circular) — plausible but wrong for this question
        mismatch_answer = str(all_samples[(i + n // 2 + 1) % len(all_samples)].get("answer", ""))

        if verbose:
            print(f"  [{i+1}/{n}] sample {sample_id} — judging gold / mismatch / refusal...")

        gold_result = _judge_tier(task, question, gold_answer, gold_answer, "gold", model, sample_id, delay)
        mismatch_result = _judge_tier(task, question, mismatch_answer, gold_answer, "mismatch", model, sample_id, delay)
        refusal_result = _judge_tier(task, question, _REFUSAL_TEXT, gold_answer, "refusal", model, sample_id, delay)

        gs = gold_result.normalized_score
        ms = mismatch_result.normalized_score
        rs = refusal_result.normalized_score
        gold_beats_bad = gs > max(ms, rs)

        if verbose:
            icon = "✓" if gold_beats_bad else "✗"
            print(f"    gold={gs:.1f}  mismatch={ms:.1f}  refusal={rs:.1f}  {icon}")

        gold_scores.append(gs)
        mismatch_scores.append(ms)
        refusal_scores.append(rs)
        sample_alignments.append(SampleAlignment(
            sample_id=sample_id,
            task=task,
            gold_score=gs,
            mismatch_score=ms,
            refusal_score=rs,
            gold_beats_bad=gold_beats_bad,
        ))

    # discrimination = % of samples where gold beats both bad tiers
    discrimination = sum(s.gold_beats_bad for s in sample_alignments) / n

    # Spearman rho: expected tier order (gold=3, mismatch=2, refusal=1) vs actual scores
    # Flatten all three tiers into paired lists
    expected_flat = [3.0] * n + [2.0] * n + [1.0] * n
    actual_flat = gold_scores + mismatch_scores + refusal_scores
    rho = _spearman_rho(expected_flat, actual_flat)

    return TaskAlignment(
        task=task,
        n_samples=n,
        ceiling_score=round(sum(gold_scores) / n, 2),
        mismatch_score=round(sum(mismatch_scores) / n, 2),
        refusal_score=round(sum(refusal_scores) / n, 2),
        discrimination_rate=round(discrimination, 4),
        spearman_rho=rho,
        samples=sample_alignments,
    )


# ── Mode 2: Real-model comparison ────────────────────────────────────────────

@dataclass
class ModelCompareSample:
    sample_id: int
    task: str
    gold_score: float
    model_score: float
    gold_wins: bool   # gold_score > model_score


@dataclass
class ModelCompareTask:
    task: str
    n_samples: int
    model_name: str
    expected_model_score: float   # externally known human score for the DUT
    ceiling_score: float
    model_score: float
    gap: float                    # ceiling − model_score
    expected_gap: float           # expected from external scores (95 − expected_model_score)
    discrimination_rate: float    # % of samples where gold > model
    spearman_rho: float
    samples: list[ModelCompareSample] = field(default_factory=list)


def compare_task(
    task: str,
    benchmark_dir: Path,
    compare_dir: Path,
    model_name: str,
    expected_model_score: float,
    n_samples: int,
    judge_model: str,
    delay: float,
    verbose: bool,
    calibrate_n: int = 0,
    calibrate_mode: str = "random",
) -> ModelCompareTask:
    """Judge gold vs real DUT responses for the same questions, matched by sample ID."""
    # Load gold samples
    gold_samples = {
        s["other"]["id"]: s
        for s in _load_samples(benchmark_dir, task)
        if isinstance(s.get("other"), dict) and "id" in s["other"]
    }
    # Load DUT samples
    dut_samples = {
        s["other"]["id"]: s
        for s in _load_samples(compare_dir, task)
        if isinstance(s.get("other"), dict) and "id" in s["other"]
    }

    # Match by ID
    common_ids = sorted(set(gold_samples) & set(dut_samples))
    if not common_ids:
        raise ValueError(f"{task}: no matching sample IDs between benchmark and compare-dir")

    selected_ids = common_ids[:n_samples]
    selected_id_set = set(selected_ids)

    # Set up calibration: build RAG index or load static anchor pool once per task
    rag_index: GoldRAG | None = None
    anchor_examples_static: list[dict] | None = None
    if calibrate_n > 0:
        if calibrate_mode == "random":
            anchor_examples_static = _load_anchor_examples(
                benchmark_dir, task, n=calibrate_n, seed=99, exclude_ids=selected_id_set
            )
            if verbose and anchor_examples_static:
                print(f"  [{task}] calibrating with {len(anchor_examples_static)} gold anchors (random)")
        else:
            try:
                rag_index = GoldRAG(benchmark_dir, task, backend=calibrate_mode)
                if verbose:
                    print(f"  [{task}] RAG index built ({calibrate_mode}, {len(rag_index)} samples)")
            except Exception as e:
                if verbose:
                    print(f"  [{task}] RAG init failed ({e}), falling back to random anchors")
                anchor_examples_static = _load_anchor_examples(
                    benchmark_dir, task, n=calibrate_n, seed=99, exclude_ids=selected_id_set
                )

    gold_scores, model_scores = [], []
    compare_samples: list[ModelCompareSample] = []

    for sample_id in selected_ids:
        gold_s = gold_samples[sample_id]
        dut_s = dut_samples[sample_id]
        question = str(gold_s.get("question", ""))
        gold_answer = str(gold_s.get("answer", ""))
        model_answer = str(dut_s.get("answer", ""))

        # Resolve per-sample anchor examples
        anchor_examples: list[dict] | None = None
        if calibrate_n > 0:
            if rag_index is not None:
                anchor_examples = rag_index.retrieve(
                    question=question,
                    n=calibrate_n,
                    exclude_ids={int(sample_id)},
                )
            else:
                anchor_examples = anchor_examples_static

        if verbose:
            print(f"  [id={sample_id}] judging gold / {model_name}...")

        # Pass anchor_examples into the judge via judge_response directly
        from judge.judge import judge_response as _jr
        try:
            gr = _jr(task, question, gold_answer, gold_answer, judge_model, "gold-calibrate", anchor_examples)
            gold_score = gr.normalized_score
        except Exception:
            gold_score = 0.0
        import time as _time
        _time.sleep(delay)
        try:
            mr = _jr(task, question, model_answer, gold_answer, judge_model, model_name, anchor_examples)
            model_score = mr.normalized_score
        except Exception:
            model_score = 0.0
        _time.sleep(delay)

        gs, ms = gold_score, model_score
        gold_wins = gs > ms

        if verbose:
            icon = "✓" if gold_wins else "✗"
            print(f"    gold={gs:.1f}  {model_name}={ms:.1f}  gap={gs-ms:+.1f}  {icon}")

        gold_scores.append(gs)
        model_scores.append(ms)
        compare_samples.append(ModelCompareSample(
            sample_id=sample_id, task=task,
            gold_score=gs, model_score=ms, gold_wins=gold_wins,
        ))

    n = len(compare_samples)
    avg_gold = sum(gold_scores) / n
    avg_model = sum(model_scores) / n
    discrimination = sum(s.gold_wins for s in compare_samples) / n
    expected_gap = 95.0 - expected_model_score

    # Spearman rho: expected [gold=2, model=1] vs actual scores
    expected_flat = [2.0] * n + [1.0] * n
    actual_flat = gold_scores + model_scores
    rho = _spearman_rho(expected_flat, actual_flat)

    return ModelCompareTask(
        task=task,
        n_samples=n,
        model_name=model_name,
        expected_model_score=expected_model_score,
        ceiling_score=round(avg_gold, 2),
        model_score=round(avg_model, 2),
        gap=round(avg_gold - avg_model, 2),
        expected_gap=round(expected_gap, 2),
        discrimination_rate=round(discrimination, 4),
        spearman_rho=rho,
        samples=compare_samples,
    )


def _write_compare_tsv(out_dir: Path, compares: list[ModelCompareTask]) -> None:
    tsv_path = out_dir / "compare-results.tsv"
    rows = ["task\tsample_id\tgold_score\tmodel_score\tgap\tgold_wins"]
    for ct in compares:
        for s in ct.samples:
            rows.append(
                f"{s.task}\t{s.sample_id}\t{s.gold_score:.1f}\t"
                f"{s.model_score:.1f}\t{s.gold_score - s.model_score:+.1f}\t"
                f"{'yes' if s.gold_wins else 'no'}"
            )
    tsv_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"\nSaved: {tsv_path}")


def _write_compare_report(
    out_dir: Path,
    compares: list[ModelCompareTask],
    judge_model: str,
) -> None:
    model_name = compares[0].model_name if compares else "DUT"
    expected_score = compares[0].expected_model_score if compares else 0
    expected_gap = compares[0].expected_gap if compares else 0

    lines = [
        f"# Judge vs Human: Gold vs {model_name}",
        f"\nJudge: `{judge_model}`  |  DUT: `{model_name}` (human score ≈ {expected_score})  "
        f"|  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## What this tests",
        f"The human evaluation established: gold ≈ 95, {model_name} ≈ {expected_score:.0f} "
        f"(gap ≈ {expected_gap:.0f} pts).",
        "This report checks whether the LLM-as-judge reproduces that gap.",
        "",
        "- **Discrimination rate**: % of samples where judge scores gold > DUT  (target ≥ 80%)",
        "- **Gap alignment**: judge gap vs expected gap — values close to 1.0x are ideal",
        "- **Spearman ρ**: rank correlation gold vs DUT order  (target ≥ 0.7)",
        "",
        "## Per-Task Results",
        "",
        f"| Task | n | Gold score | {model_name} score | Actual gap | Expected gap | Gap ratio | Discrimination | ρ | Status |",
        f"|------|---|-----------|{'-' * (len(model_name)+8)}|-----------|-------------|-----------|----------------|---|--------|",
    ]

    overall_disc, overall_rho, overall_gap, overall_ceil, overall_model = [], [], [], [], []

    for ct in compares:
        gap_ratio = ct.gap / ct.expected_gap if ct.expected_gap else 0
        disc_ok = ct.discrimination_rate >= 0.80
        rho_ok = ct.spearman_rho >= 0.70
        gap_ok = 0.5 <= gap_ratio <= 1.5   # judge gap within 50% of expected

        status = "PASS" if (disc_ok and rho_ok) else "REVIEW"
        flags = []
        if not disc_ok:
            flags.append(f"disc {ct.discrimination_rate:.0%}<80%")
        if not rho_ok:
            flags.append(f"ρ {ct.spearman_rho:.2f}<0.7")
        if not gap_ok:
            flags.append(f"gap ratio {gap_ratio:.1f}x")

        status_cell = f"{'✓' if status == 'PASS' else '⚠'} {status}"
        if flags:
            status_cell += f" ({'; '.join(flags)})"

        lines.append(
            f"| {ct.task} | {ct.n_samples} | {ct.ceiling_score:.1f} | "
            f"{ct.model_score:.1f} | {ct.gap:+.1f} | {ct.expected_gap:+.1f} | "
            f"{gap_ratio:.2f}x | {ct.discrimination_rate:.0%} | {ct.spearman_rho:.2f} | {status_cell} |"
        )
        overall_disc.append(ct.discrimination_rate)
        overall_rho.append(ct.spearman_rho)
        overall_gap.append(ct.gap)
        overall_ceil.append(ct.ceiling_score)
        overall_model.append(ct.model_score)

    if compares:
        avg_disc = sum(overall_disc) / len(compares)
        avg_rho = sum(overall_rho) / len(compares)
        avg_gap = sum(overall_gap) / len(compares)
        avg_ceil = sum(overall_ceil) / len(compares)
        avg_model_s = sum(overall_model) / len(compares)
        avg_gap_ratio = avg_gap / expected_gap if expected_gap else 0
        overall_status = "PASS" if avg_disc >= 0.80 and avg_rho >= 0.70 else "REVIEW"
        lines.append(
            f"| **Overall** | — | **{avg_ceil:.1f}** | **{avg_model_s:.1f}** | "
            f"**{avg_gap:+.1f}** | **{expected_gap:+.1f}** | **{avg_gap_ratio:.2f}x** | "
            f"**{avg_disc:.0%}** | **{avg_rho:.2f}** | "
            f"{'✓' if overall_status == 'PASS' else '⚠'} **{overall_status}** |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "**Gap ratio > 1.5x**: judge exaggerates quality difference → rubric anchors too extreme.",
        "**Gap ratio < 0.5x**: judge can't distinguish gold from DUT → rubric anchors need sharpening.",
        "**Discrimination < 80%**: judge ranks DUT above gold on >20% of samples → calibration needed.",
        "",
        "To optimize: run `/autoresearch` with metric=discrimination_rate, scope=judge/rubrics.py+judge/judge.py",
    ]

    report_path = out_dir / "compare-report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")


def _write_tsv(out_dir: Path, alignments: list[TaskAlignment]) -> None:
    tsv_path = out_dir / "validate-results.tsv"
    rows = [
        "task\tsample_id\tgold_score\tmismatch_score\trefusal_score\tgold_beats_bad"
    ]
    for ta in alignments:
        for sa in ta.samples:
            rows.append(
                f"{sa.task}\t{sa.sample_id}\t{sa.gold_score:.1f}\t"
                f"{sa.mismatch_score:.1f}\t{sa.refusal_score:.1f}\t"
                f"{'yes' if sa.gold_beats_bad else 'no'}"
            )
    tsv_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"\nSaved: {tsv_path}")


def _write_report(out_dir: Path, alignments: list[TaskAlignment], model: str) -> None:
    lines = [
        "# Judge-Human Alignment Validation Report",
        f"\nJudge model: `{model}`  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## How to read this report",
        "",
        "| Tier | What it is | Expected score |",
        "|------|-----------|----------------|",
        "| GOLD | Benchmark answer (human-curated) | ≥ 85/100 |",
        "| MISMATCH | Different sample's answer (wrong for this Q) | ≤ 55/100 |",
        "| REFUSAL | 'Cannot determine.' (off-task) | ≤ 20/100 |",
        "",
        "**Discrimination rate**: % of samples where gold > max(mismatch, refusal)  (target ≥ 80%)",
        "**Spearman ρ**: rank correlation between expected tier order and actual scores (target ≥ 0.7)",
        "",
        "## Per-Task Results",
        "",
        "| Task | n | Ceiling (Gold) | Mismatch | Refusal | Discrimination | Spearman ρ | Status |",
        "|------|---|---------------|---------|---------|----------------|-----------|--------|",
    ]

    overall_discrimination = []
    overall_rho = []

    for ta in alignments:
        ceiling_ok = ta.ceiling_score >= 85
        mismatch_ok = ta.mismatch_score <= 55
        refusal_ok = ta.refusal_score <= 20
        disc_ok = ta.discrimination_rate >= 0.80
        rho_ok = ta.spearman_rho >= 0.70

        status = "PASS" if (ceiling_ok and disc_ok and rho_ok) else "REVIEW"
        flags = []
        if not ceiling_ok:
            flags.append(f"ceiling low ({ta.ceiling_score:.1f}<85)")
        if not mismatch_ok:
            flags.append(f"mismatch high ({ta.mismatch_score:.1f}>55)")
        if not refusal_ok:
            flags.append(f"refusal high ({ta.refusal_score:.1f}>20)")
        if not disc_ok:
            flags.append(f"discrimination low ({ta.discrimination_rate:.0%}<80%)")
        if not rho_ok:
            flags.append(f"ρ low ({ta.spearman_rho:.2f}<0.7)")

        status_cell = f"{'✓' if status == 'PASS' else '⚠'} {status}"
        if flags:
            status_cell += f" ({'; '.join(flags)})"

        lines.append(
            f"| {ta.task} | {ta.n_samples} | {ta.ceiling_score:.1f} | "
            f"{ta.mismatch_score:.1f} | {ta.refusal_score:.1f} | "
            f"{ta.discrimination_rate:.0%} | {ta.spearman_rho:.2f} | {status_cell} |"
        )
        overall_discrimination.append(ta.discrimination_rate)
        overall_rho.append(ta.spearman_rho)

    n_tasks = len(alignments)
    if n_tasks > 0:
        avg_disc = sum(overall_discrimination) / n_tasks
        avg_rho = sum(overall_rho) / n_tasks
        avg_ceil = sum(ta.ceiling_score for ta in alignments) / n_tasks
        overall_status = "PASS" if avg_disc >= 0.80 and avg_rho >= 0.70 else "REVIEW"
        lines += [
            f"| **Overall** | — | **{avg_ceil:.1f}** | — | — | "
            f"**{avg_disc:.0%}** | **{avg_rho:.2f}** | "
            f"{'✓' if overall_status == 'PASS' else '⚠'} **{overall_status}** |",
        ]

    lines += [
        "",
        "## Optimization Guide",
        "",
        "If **ceiling < 85**: the judge is too strict or rubric anchors don't match the gold answers.",
        "  → Lower score_5 bar for criteria OR soften the SYSTEM_PROMPT scoring philosophy.",
        "",
        "If **mismatch > 55**: the judge can't tell a wrong answer from a correct one.",
        "  → Sharpen rubric `score_1` / `score_5` anchors for the failing task.",
        "  → Add explicit discriminating language to the task's rubric description.",
        "",
        "If **refusal > 20**: the judge rewards unhelpful responses.",
        "  → Add to SYSTEM_PROMPT: 'Refusing to answer when a clinical answer is required scores 1.'",
        "",
        "If **discrimination < 80% or ρ < 0.7**: systemic alignment failure.",
        "  → Run `/autoresearch` with goal='maximize validate.py alignment score' and",
        "    scope='judge/judge.py SYSTEM_PROMPT and judge/rubrics.py criteria anchors'.",
    ]

    report_path = out_dir / "validate-report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate LLM-as-judge alignment with human gold answers"
    )
    parser.add_argument("--benchmark", default="medbench-agent-95", help="Benchmark directory")
    parser.add_argument("--task", help="Single task to validate (e.g. MedCOT)")
    parser.add_argument("--all-tasks", action="store_true", help="Validate all 13 tasks")
    parser.add_argument("--samples", type=int, default=5, help="Samples per task (default 5)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Judge model")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--output-dir", default="validate", help="Output directory base")
    parser.add_argument("--verbose", action="store_true", default=True)
    # Mode 2: real model comparison
    parser.add_argument(
        "--compare-dir",
        help="Directory with real DUT responses (same JSONL schema as benchmark). Enables Mode 2.",
    )
    parser.add_argument(
        "--compare-name", default="dut",
        help="Human-readable name for the DUT (e.g. gpt-4.1)",
    )
    parser.add_argument(
        "--compare-expected-score", type=float, default=81.0,
        help="Externally known human score for the DUT (default 81). Used to compute expected gap.",
    )
    parser.add_argument(
        "--calibrate-n", type=int, default=0, metavar="N",
        help="Show N gold anchor examples to the judge (few-shot calibration). Default: 0 (off).",
    )
    parser.add_argument(
        "--calibrate-mode",
        choices=["random", "bm25", "embedding"],
        default="random",
        help=(
            "How to select calibration anchor examples. "
            "'random' (default) uses a fixed seed per task. "
            "'bm25' or 'embedding' use semantic retrieval (GoldRAG) to pick "
            "the most similar gold examples per question. "
            "Requires --calibrate-n > 0."
        ),
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark)
    if not benchmark_dir.exists():
        raise SystemExit(f"Benchmark directory not found: {benchmark_dir}")

    if args.all_tasks:
        tasks = list_tasks()
    elif args.task:
        tasks = [args.task]
    else:
        raise SystemExit("Specify --task <name> or --all-tasks")

    slug = datetime.now().strftime("%y%m%d-%H%M")

    # ── Mode 2: real model comparison ────────────────────────────────────────
    if args.compare_dir:
        compare_dir = Path(args.compare_dir)
        if not compare_dir.exists():
            raise SystemExit(f"compare-dir not found: {compare_dir}")

        out_dir = Path(args.output_dir) / f"{slug}-compare-{args.compare_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print("Judge vs Human: Gold vs DUT Comparison")
        print(f"  DUT: {args.compare_name} (human score ≈ {args.compare_expected_score})")
        print(f"  Tasks: {', '.join(tasks)}")
        print(f"  Samples per task: {args.samples}")
        print(f"  Judge model: {args.model}")
        print(f"  Output: {out_dir}/\n")

        compares: list[ModelCompareTask] = []
        for task in tasks:
            print(f"\n[{task}]")
            try:
                ct = compare_task(
                    task=task,
                    benchmark_dir=benchmark_dir,
                    compare_dir=compare_dir,
                    model_name=args.compare_name,
                    expected_model_score=args.compare_expected_score,
                    n_samples=args.samples,
                    judge_model=args.model,
                    delay=args.delay,
                    verbose=args.verbose,
                    calibrate_n=args.calibrate_n,
                    calibrate_mode=args.calibrate_mode,
                )
                compares.append(ct)
                print(
                    f"  gold={ct.ceiling_score:.1f}  {ct.model_name}={ct.model_score:.1f}  "
                    f"gap={ct.gap:+.1f} (expected {ct.expected_gap:+.1f})  "
                    f"disc={ct.discrimination_rate:.0%}  ρ={ct.spearman_rho:.2f}"
                )
            except Exception as e:
                print(f"  ERROR: {e}")

        if compares:
            _write_compare_tsv(out_dir, compares)
            _write_compare_report(out_dir, compares, judge_model=args.model)

        print("\nDone.")
        return

    # ── Mode 1: synthetic tier test ───────────────────────────────────────────
    out_dir = Path(args.output_dir) / f"{slug}-alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Judge-Human Alignment Validation (synthetic tiers)")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"  Samples per task: {args.samples}")
    print(f"  Judge model: {args.model}")
    print(f"  Output: {out_dir}/\n")

    alignments: list[TaskAlignment] = []
    for task in tasks:
        print(f"\n[{task}]")
        try:
            ta = validate_task(
                task=task,
                benchmark_dir=benchmark_dir,
                n_samples=args.samples,
                model=args.model,
                delay=args.delay,
                verbose=args.verbose,
            )
            alignments.append(ta)
            print(
                f"  ceiling={ta.ceiling_score:.1f}  mismatch={ta.mismatch_score:.1f}  "
                f"refusal={ta.refusal_score:.1f}  "
                f"discrimination={ta.discrimination_rate:.0%}  ρ={ta.spearman_rho:.2f}"
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    if alignments:
        _write_tsv(out_dir, alignments)
        _write_report(out_dir, alignments, model=args.model)

    print("\nDone.")


if __name__ == "__main__":
    main()
