"""
Report generation for MedBench-Agent-95 evaluation results.
Outputs JSON summary and markdown report.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .batch_runner import BenchmarkResult, SampleResult


def _task_stats(results: list[SampleResult]) -> dict[str, Any]:
    """Compute per-task statistics."""
    if not results:
        return {"n": 0, "avg_score": 0.0, "min_score": 0.0, "max_score": 0.0, "errors": 0,
                "total_minor_errors": 0, "total_major_errors": 0}

    scores = [r.judgement.normalized_score for r in results if not r.judgement.error]
    errors = sum(1 for r in results if r.judgement.error)

    # Per-criterion averages
    criterion_avgs: dict[str, list[float]] = {}
    for r in results:
        for cs in r.judgement.criterion_scores:
            criterion_avgs.setdefault(cs.name, []).append(cs.score)

    criterion_means = {
        k: round(sum(v) / len(v), 3) for k, v in criterion_avgs.items()
    }

    total_minor = sum(len(r.judgement.minor_errors) for r in results)
    total_major = sum(len(r.judgement.major_errors) for r in results)

    return {
        "n": len(results),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min_score": round(min(scores), 2) if scores else 0.0,
        "max_score": round(max(scores), 2) if scores else 0.0,
        "errors": errors,
        "total_minor_errors": total_minor,
        "total_major_errors": total_major,
        "criterion_means": criterion_means,
    }


def build_summary(result: BenchmarkResult) -> dict[str, Any]:
    """Build a structured summary dict from benchmark results."""
    task_stats = {}
    all_scores = []

    for task, results in result.task_results.items():
        stats = _task_stats(results)
        task_stats[task] = stats
        if stats["n"] > 0:
            all_scores.extend(
                r.judgement.normalized_score
                for r in results
                if not r.judgement.error
            )

    overall_avg = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0

    # Collect DUT name from first available result
    dut = "unknown"
    for task_results in result.task_results.values():
        for r in task_results:
            if r.judgement.dut:
                dut = r.judgement.dut
                break
        if dut != "unknown":
            break

    return {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "judge_model": result.model,
            "dut": dut,
            "benchmark_dir": result.benchmark_dir,
            "tasks_evaluated": list(result.task_results.keys()),
            "total_samples": sum(s["n"] for s in task_stats.values()),
            "total_errors": sum(s["errors"] for s in task_stats.values()),
            "total_minor_errors": sum(s.get("total_minor_errors", 0) for s in task_stats.values()),
            "total_major_errors": sum(s.get("total_major_errors", 0) for s in task_stats.values()),
        },
        "overall": {
            "avg_score": overall_avg,
            "score_range": "0-100 (normalized from 1-5 Likert)",
        },
        "tasks": task_stats,
        "errors": result.errors,
    }


def save_results(result: BenchmarkResult, output_dir: str) -> dict[str, str]:
    """
    Save evaluation results to output_dir.
    Returns dict of {filename: path} for generated files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = build_summary(result)

    # 1. Summary JSON
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Detailed results JSONL
    details_path = out / "details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for task, task_results in result.task_results.items():
            for r in task_results:
                record = {
                    "task": r.task,
                    "sample_id": r.sample_id,
                    "source": r.source,
                    "dut": r.judgement.dut,
                    "is_gold_eval": r.is_gold_eval,
                    "normalized_score": r.judgement.normalized_score,
                    "total_score": r.judgement.total_score,
                    "criteria": [
                        {"name": cs.name, "score": cs.score, "justification": cs.justification}
                        for cs in r.judgement.criterion_scores
                    ],
                    "overall_feedback": r.judgement.overall_feedback,
                    "minor_errors": r.judgement.minor_errors,
                    "major_errors": r.judgement.major_errors,
                    "error": r.judgement.error,
                    "tokens": {
                        "input": r.judgement.input_tokens,
                        "output": r.judgement.output_tokens,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 3. Markdown report
    md_path = out / "report.md"
    md_path.write_text(_build_markdown(summary, result), encoding="utf-8")

    return {
        "summary": str(summary_path),
        "details": str(details_path),
        "report": str(md_path),
    }


def _build_markdown(summary: dict[str, Any], result: BenchmarkResult) -> str:
    meta = summary["meta"]
    overall = summary["overall"]
    tasks = summary["tasks"]

    lines = [
        "# MedBench-Eval Report",
        "",
        f"**Timestamp:** {meta['timestamp']}  ",
        f"**DUT (Model Under Test):** `{meta['dut']}`  ",
        f"**Judge Model:** `{meta['judge_model']}`  ",
        f"**Benchmark:** `{meta['benchmark_dir']}`  ",
        f"**Total Samples:** {meta['total_samples']}  ",
        f"**Total Errors:** {meta['total_errors']}  ",
        f"**Minor Errors (no score impact):** {meta.get('total_minor_errors', 0)}  ",
        f"**Major Errors (score impact):** {meta.get('total_major_errors', 0)}",
        "",
        "## Overall",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Average Score | **{overall['avg_score']}/100** |",
        f"| Score Range | {overall['score_range']} |",
        "",
        "## Per-Task Results",
        "",
        "| Task | N | Avg Score | Min | Max | Minor Err | Major Err | Judge Err |",
        "|------|---|-----------|-----|-----|-----------|-----------|-----------|",
    ]

    for task, stats in sorted(tasks.items()):
        lines.append(
            f"| {task} | {stats['n']} | {stats['avg_score']:.1f} | "
            f"{stats['min_score']:.1f} | {stats['max_score']:.1f} | "
            f"{stats.get('total_minor_errors', 0)} | "
            f"{stats.get('total_major_errors', 0)} | "
            f"{stats['errors']} |"
        )

    lines += ["", "## Per-Criterion Breakdown", ""]

    for task, stats in sorted(tasks.items()):
        if not stats.get("criterion_means"):
            continue
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| Criterion | Avg Score (1-5) |")
        lines.append("|-----------|-----------------|")
        for crit, mean in stats["criterion_means"].items():
            lines.append(f"| {crit} | {mean:.2f} |")
        lines.append("")

    if result.errors:
        lines += ["## Errors", ""]
        for err in result.errors:
            lines.append(f"- {err}")
        lines.append("")

    lines += [
        "---",
        "*Generated by [medbench-eval](https://github.com/chenhao/medqa-auto-eval-skills)*",
    ]

    return "\n".join(lines)


def print_summary(result: BenchmarkResult) -> None:
    """Print a concise summary to stdout."""
    summary = build_summary(result)
    overall = summary["overall"]
    tasks = summary["tasks"]

    print("\n" + "=" * 60)
    print("MedBench-Eval Results")
    print("=" * 60)
    print(f"Judge model : {result.model}")
    print(f"Overall avg : {overall['avg_score']:.1f}/100")
    print()
    print(f"{'Task':<18} {'N':>4} {'Avg':>8} {'Min':>8} {'Max':>8} {'Err':>5}")
    print("-" * 60)
    for task, stats in sorted(tasks.items()):
        print(
            f"{task:<18} {stats['n']:>4} "
            f"{stats['avg_score']:>8.1f} {stats['min_score']:>8.1f} "
            f"{stats['max_score']:>8.1f} {stats['errors']:>5}"
        )
    print("=" * 60)
