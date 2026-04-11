"""
Batch evaluation runner for MedBench-Agent-95.
Loads benchmark JSONL files and runs the LLM-as-judge over samples.
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .judge import judge_response, judge_against_gold, JudgementResult
from .rubrics import list_tasks, RUBRICS
from .capabilities import get_tasks_for_capability, CAPABILITY_GROUPS
from .models import DEFAULT_MODEL


BENCHMARK_TASKS = list_tasks()  # all 13 tasks


@dataclass
class SampleResult:
    task: str
    sample_id: int
    source: str
    question: str
    gold_answer: str
    evaluated_response: str   # the response being judged (could be gold or a model output)
    judgement: JudgementResult
    is_gold_eval: bool = False  # True when evaluating gold answer itself


@dataclass
class BenchmarkResult:
    task_results: dict[str, list[SampleResult]] = field(default_factory=dict)
    model: str = DEFAULT_MODEL
    benchmark_dir: str = ""
    errors: list[str] = field(default_factory=list)


def _load_task_samples(benchmark_dir: Path, task: str) -> list[dict]:
    """Load JSONL samples for a task. Returns list of dicts with question/answer/other."""
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


def run_benchmark(
    benchmark_dir: str,
    tasks: Optional[list[str]] = None,
    capability: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    dut: str = "unknown",
    samples_per_task: int = 5,
    evaluate_gold: bool = False,
    responses_dir: Optional[str] = None,
    seed: int = 42,
    delay_seconds: float = 0.5,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run the LLM-as-judge over benchmark samples.

    Args:
        benchmark_dir: Path to the medbench-agent-95/ directory
        tasks: List of task names to evaluate (None = all tasks)
        capability: Capability group key to evaluate (overrides tasks if set)
        model: Judge model to use
        dut: Name of the model/system under test (for reporting)
        samples_per_task: Number of samples to evaluate per task
        evaluate_gold: If True, evaluate gold answers (ceiling calibration)
        responses_dir: If provided, load model responses from this directory
                       Expected: {responses_dir}/{task}/{sample_id}.txt
        seed: Random seed for sample selection
        delay_seconds: Delay between API calls to avoid rate limiting
        verbose: Print progress to stdout

    Returns:
        BenchmarkResult with all sample results
    """
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        raise ValueError(f"Benchmark directory not found: {benchmark_dir}")

    # Resolve tasks: capability group takes priority, then explicit tasks, then all
    if capability:
        tasks_to_run = list(get_tasks_for_capability(capability))
    else:
        tasks_to_run = tasks or BENCHMARK_TASKS
    # Filter to only tasks that have JSONL files in the benchmark dir
    tasks_to_run = [t for t in tasks_to_run if (benchmark_path / f"{t}.jsonl").exists()]

    result = BenchmarkResult(model=model, benchmark_dir=str(benchmark_path))
    rng = random.Random(seed)

    for task in tasks_to_run:
        if verbose:
            print(f"\n[{task}] Loading samples...")

        samples = _load_task_samples(benchmark_path, task)
        if not samples:
            result.errors.append(f"{task}: no samples found")
            continue

        selected = rng.sample(samples, min(samples_per_task, len(samples)))
        task_results = []

        for i, sample in enumerate(selected):
            question = str(sample.get("question", ""))
            gold_answer = str(sample.get("answer", ""))
            other = sample.get("other", {})
            sample_id = other.get("id", i) if isinstance(other, dict) else i
            source = other.get("source", task) if isinstance(other, dict) else task

            # Determine what response to evaluate
            if evaluate_gold:
                evaluated_response = gold_answer
                is_gold = True
            elif responses_dir:
                resp_path = Path(responses_dir) / task / f"{sample_id}.txt"
                if resp_path.exists():
                    evaluated_response = resp_path.read_text(encoding="utf-8")
                else:
                    result.errors.append(f"{task}/{sample_id}: response file not found at {resp_path}")
                    continue
                is_gold = False
            else:
                # Default: evaluate gold answer as calibration baseline
                evaluated_response = gold_answer
                is_gold = True

            if verbose:
                mode = "gold" if is_gold else "model"
                print(f"  [{i+1}/{len(selected)}] {task} sample {sample_id} ({mode})...")

            if is_gold:
                judgement = judge_against_gold(
                    task=task,
                    question=question,
                    gold_answer=gold_answer,
                    model=model,
                    dut=dut,
                )
            else:
                judgement = judge_response(
                    task=task,
                    question=question,
                    response=evaluated_response,
                    gold_answer=gold_answer,
                    model=model,
                    dut=dut,
                )

            if judgement.error and verbose:
                print(f"    ERROR: {judgement.error}")

            task_results.append(SampleResult(
                task=task,
                sample_id=sample_id,
                source=source,
                question=question,
                gold_answer=gold_answer,
                evaluated_response=evaluated_response,
                judgement=judgement,
                is_gold_eval=is_gold,
            ))

            if delay_seconds > 0 and i < len(selected) - 1:
                time.sleep(delay_seconds)

        result.task_results[task] = task_results
        if verbose and task_results:
            avg = sum(r.judgement.normalized_score for r in task_results) / len(task_results)
            print(f"  [{task}] avg score: {avg:.1f}/100")

    return result
