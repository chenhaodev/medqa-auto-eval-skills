#!/usr/bin/env python3
"""
medbench-eval: LLM-as-judge for MedBench-Agent-95 medical AI benchmark.

Usage:
  # Interactive wizard (recommended — prompts for capability, DUT, options)
  python eval.py

  # Evaluate a specific capability group
  python eval.py batch --benchmark medbench-agent-95/ --capability reasoning

  # Evaluate a specific task only
  python eval.py batch --benchmark medbench-agent-95/ --task MedCOT

  # Evaluate model responses from a directory
  python eval.py batch --benchmark medbench-agent-95/ --responses-dir my_model_outputs/ --dut gpt-4o

  # Judge single response
  python eval.py single --task MedCOT --question "..." --response "..." --dut my-model

  # Use minimax model as judge
  python eval.py batch --benchmark medbench-agent-95/ --model minimax-m2.5

Environment variables:
  ANTHROPIC_API_KEY   Required for claude-haiku-4-5 (default judge model)
  MINIMAX_API_KEY     Required for minimax-m2.5
"""

import argparse
import json
import sys
from pathlib import Path


def _interactive_wizard() -> argparse.Namespace:
    """
    Interactive session wizard. Prompts user for capability, DUT, and options.
    Returns a populated Namespace as if batch command args were parsed.
    """
    from judge.capabilities import list_capabilities

    print()
    print("=" * 60)
    print("  medbench-eval — Interactive Session Wizard")
    print("=" * 60)
    print()

    # Step 1: DUT identification
    print("Step 1/4 — What model/system are you evaluating? (DUT)")
    print("  Examples: gpt-4o, claude-opus-4-6, my-fine-tuned-model, ...")
    dut = input("  DUT name [unknown]: ").strip() or "unknown"
    print()

    # Step 2: Capability selection
    caps = list_capabilities()
    print("Step 2/4 — Which capability would you like to test?")
    print()
    for i, cap in enumerate(caps, 1):
        tasks_str = ", ".join(cap.tasks)
        print(f"  [{i}] {cap.name}")
        print(f"      {cap.description}")
        print(f"      Tasks: {tasks_str}")
        if cap.example_question:
            print(f"      Example: \"{cap.example_question[:80]}\"")
        print()

    while True:
        choice = input(f"  Select [1-{len(caps)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(caps):
            selected_cap = caps[int(choice) - 1]
            break
        print(f"  Please enter a number between 1 and {len(caps)}")

    print(f"\n  Selected: {selected_cap.name} ({', '.join(selected_cap.tasks)})")
    print()

    # Step 3: Samples per task
    print("Step 3/4 — How many samples per task?")
    print("  [1] Quick (3 samples)   [2] Standard (5)   [3] Thorough (10)   [4] Full (30)")
    sample_map = {"1": 3, "2": 5, "3": 10, "4": 30}
    sample_choice = input("  Select [1-4, default=2]: ").strip() or "2"
    samples = sample_map.get(sample_choice, 5)
    print(f"  Samples per task: {samples}")
    print()

    # Step 4: Benchmark dir + output
    print("Step 4/4 — Paths")
    benchmark_dir = input("  Benchmark dir [medbench-agent-95/]: ").strip() or "medbench-agent-95/"
    output_dir = input("  Output dir [results/]: ").strip() or "results/"
    responses_dir = input("  Model responses dir (leave blank to use gold answers): ").strip() or None
    print()

    # Confirm
    print("─" * 60)
    print(f"  DUT            : {dut}")
    print(f"  Capability     : {selected_cap.name}")
    print(f"  Tasks          : {', '.join(selected_cap.tasks)}")
    print(f"  Samples/task   : {samples}")
    print(f"  Benchmark dir  : {benchmark_dir}")
    print(f"  Responses dir  : {responses_dir or '(using gold answers as baseline)'}")
    print(f"  Output dir     : {output_dir}")
    print("─" * 60)
    confirm = input("\n  Start evaluation? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("Cancelled.")
        sys.exit(0)
    print()

    # Build a fake Namespace
    ns = argparse.Namespace(
        command="batch",
        benchmark=benchmark_dir,
        capability=selected_cap.key,
        task=None,
        dut=dut,
        model="claude-haiku-4-5",
        responses_dir=responses_dir,
        samples=samples,
        output=output_dir,
        seed=42,
        delay=0.5,
        quiet=False,
    )
    return ns


def cmd_batch(args: argparse.Namespace) -> None:
    """Run batch evaluation over benchmark JSONL files."""
    from judge.runner import run_benchmark
    from judge.report import save_results, print_summary

    tasks = [args.task] if getattr(args, "task", None) else None
    capability = getattr(args, "capability", None)
    dut = getattr(args, "dut", "unknown")

    result = run_benchmark(
        benchmark_dir=args.benchmark,
        tasks=tasks,
        capability=capability,
        model=args.model,
        dut=dut,
        samples_per_task=args.samples,
        evaluate_gold=not args.responses_dir,
        responses_dir=args.responses_dir,
        seed=args.seed,
        delay_seconds=args.delay,
        verbose=not args.quiet,
    )

    print_summary(result)

    if args.output:
        saved = save_results(result, args.output)
        print(f"\nResults saved to: {args.output}/")
        for name, path in saved.items():
            print(f"  {name}: {path}")


def cmd_single(args: argparse.Namespace) -> None:
    """Judge a single response from command-line arguments."""
    from judge.judge import judge_response

    if not args.question or not args.response:
        print("ERROR: --question and --response are required for single evaluation", file=sys.stderr)
        sys.exit(1)

    dut = getattr(args, "dut", "unknown")
    result = judge_response(
        task=args.task,
        question=args.question,
        response=args.response,
        gold_answer=getattr(args, "gold_answer", None) or None,
        model=args.model,
        dut=dut,
    )

    if result.error:
        print(f"ERROR: {result.error}", file=sys.stderr)
        sys.exit(1)

    output = {
        "task": result.task,
        "dut": result.dut,
        "judge_model": result.model,
        "total_score": result.total_score,
        "normalized_score": result.normalized_score,
        "criteria": [
            {"name": cs.name, "score": cs.score, "justification": cs.justification}
            for cs in result.criterion_scores
        ],
        "overall_feedback": result.overall_feedback,
        "minor_errors": result.minor_errors,
        "major_errors": result.major_errors,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_list_tasks(args: argparse.Namespace) -> None:
    """List all supported tasks and capability groups."""
    from judge.rubrics import RUBRICS
    from judge.capabilities import list_capabilities

    print("\nCapability Groups:\n")
    for cap in list_capabilities():
        tasks_str = ", ".join(cap.tasks)
        print(f"  {cap.key:<20} {cap.name}")
        print(f"  {'':20} Tasks: {tasks_str}")
        print()

    print("All Tasks:\n")
    for task, rubric in sorted(RUBRICS.items()):
        criteria = ", ".join(c.name for c in rubric.criteria)
        print(f"  {task:<18} — {len(rubric.criteria)} criteria: {criteria}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="medbench-eval: LLM-as-judge for MedBench-Agent-95",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        default="claude-haiku-4-5",
        choices=["claude-haiku-4-5", "minimax-m2.5"],
        help="Judge LLM to use (default: claude-haiku-4-5)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # batch command
    batch = subparsers.add_parser("batch", help="Batch evaluate benchmark samples")
    batch.add_argument("--benchmark", required=True, help="Path to medbench-agent-95/ directory")
    batch.add_argument(
        "--capability",
        choices=["reasoning", "long_context", "tool_use", "orchestration",
                 "self_correction", "role_adapt", "safety", "full"],
        help="Capability group to evaluate (overrides --task)",
    )
    batch.add_argument("--task", help="Evaluate a single task only (e.g. MedCOT)")
    batch.add_argument("--dut", default="unknown", help="Name of the model/system under test")
    batch.add_argument("--responses-dir", help="Directory with model responses to evaluate")
    batch.add_argument("--samples", type=int, default=5, help="Samples per task (default: 5)")
    batch.add_argument("--output", "-o", help="Output directory for results")
    batch.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    batch.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")
    batch.add_argument("--quiet", action="store_true", help="Suppress progress output")

    # single command
    single = subparsers.add_parser("single", help="Evaluate a single response")
    single.add_argument("--task", required=True, help="Task name (e.g. MedCOT)")
    single.add_argument("--question", help="The clinical question/scenario")
    single.add_argument("--response", help="The model response to evaluate")
    single.add_argument("--gold-answer", help="Optional gold standard answer for calibration")
    single.add_argument("--dut", default="unknown", help="Name of the model/system under test")

    # tasks command
    subparsers.add_parser("tasks", help="List all supported tasks and capability groups")

    return parser


def main() -> None:
    # No args at all → launch interactive wizard
    if len(sys.argv) == 1:
        args = _interactive_wizard()
        cmd_batch(args)
        return

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "batch":
        cmd_batch(args)
    elif args.command == "single":
        cmd_single(args)
    elif args.command == "tasks":
        cmd_list_tasks(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
