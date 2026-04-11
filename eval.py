#!/usr/bin/env python3
"""medbench-eval: LLM-as-judge on MedBench-Agent-95.

  python eval.py
  python eval.py generate -o generated/dut/ --task MedCOT --answer-model deepseek-chat
  python eval.py batch --capability reasoning --dut NAME ...
  python eval.py single --task MedCOT --question ... --response ... --dut NAME

Default --benchmark is references/medbench-agent-95 (gold JSONL).
Env: DEEPSEEK_API_KEY for answers; ANTHROPIC_API_KEY for default judge — see judge/llm_client.py.
"""

import argparse
import json
import sys
from pathlib import Path

from judge.refs import DEFAULT_BENCHMARK_REL, default_benchmark_dir, resolve_benchmark_dir


def _interactive_wizard() -> argparse.Namespace:
    """
    Interactive session wizard. Prompts user for capability, DUT, and options.
    Returns a populated Namespace as if batch command args were parsed.
    """
    from judge.refs import list_capabilities

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
    print("Step 4/4 — Paths & DUT responses")
    _bd = input(f"  Benchmark dir [{DEFAULT_BENCHMARK_REL}/]: ").strip()
    benchmark_dir = resolve_benchmark_dir(_bd or DEFAULT_BENCHMARK_REL)
    output_dir = input("  Output dir [results/]: ").strip() or "results/"
    print()
    print("  DUT responses input (leave blank to run gold-answer calibration):")
    print("    [1] JSONL or TXT file  — all responses in one file (--responses-file)")
    print("    [2] Directory          — per-sample .txt files    (--responses-dir)")
    print("    [blank] Use gold answers as baseline")
    resp_choice = input("  Input type [1/2/blank]: ").strip()
    responses_file: str | None = None
    responses_dir: str | None = None
    if resp_choice == "1":
        responses_file = input("  Path to JSONL/TXT file: ").strip() or None
    elif resp_choice == "2":
        responses_dir = input("  Path to responses directory: ").strip() or None
    print()

    # Confirm
    resp_display = responses_file or responses_dir or "(using gold answers as baseline)"
    print("─" * 60)
    print(f"  DUT            : {dut}")
    print(f"  Capability     : {selected_cap.name}")
    print(f"  Tasks          : {', '.join(selected_cap.tasks)}")
    print(f"  Samples/task   : {samples}")
    print(f"  Benchmark dir  : {benchmark_dir}")
    print(f"  DUT responses  : {resp_display}")
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
        responses_file=responses_file,
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
    from judge.runner import print_summary, run_benchmark, save_results

    tasks = [args.task] if getattr(args, "task", None) else None
    capability = getattr(args, "capability", None)
    dut = getattr(args, "dut", "unknown")

    responses_file = getattr(args, "responses_file", None)
    has_responses = bool(args.responses_dir or responses_file)

    result = run_benchmark(
        benchmark_dir=resolve_benchmark_dir(args.benchmark),
        tasks=tasks,
        capability=capability,
        model=args.model,
        dut=dut,
        samples_per_task=args.samples,
        evaluate_gold=not has_responses,
        responses_dir=args.responses_dir,
        responses_file=responses_file,
        calibrate_n=getattr(args, "calibrate_n", 0),
        calibrate_mode=getattr(args, "calibrate_mode", "random"),
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
    from judge.scoring import judge_response

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


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate DUT JSONL with an answer model (e.g. DeepSeek), then use batch to judge."""
    from judge.generate_dut import run_generate

    run_generate(args)


def cmd_list_tasks(args: argparse.Namespace) -> None:
    """List all supported tasks and capability groups."""
    from judge.refs import RUBRICS, list_capabilities

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
        choices=["claude-haiku-4-5", "minimax-m2.5", "deepseek-chat"],
        help="Judge LLM to use (default: claude-haiku-4-5)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # batch command
    batch = subparsers.add_parser("batch", help="Batch evaluate benchmark samples")
    batch.add_argument(
        "--benchmark",
        default=str(default_benchmark_dir()),
        help=(
            "Directory with MedBench JSONL files "
            "(default: shipped references/medbench-agent-95)"
        ),
    )
    batch.add_argument(
        "--capability",
        choices=["reasoning", "long_context", "tool_use", "orchestration",
                 "self_correction", "role_adapt", "safety", "full"],
        help="Capability group to evaluate (overrides --task)",
    )
    batch.add_argument("--task", help="Evaluate a single task only (e.g. MedCOT)")
    batch.add_argument("--dut", default="unknown", help="Name of the model/system under test")
    batch.add_argument(
        "--responses-file",
        help=(
            "Single file with all DUT responses. "
            "Accepts: compact JSONL ({task,id,response}), "
            "full benchmark JSONL ({question,answer,other}), "
            "or delimited TXT (=== MedCOT | 97 ===)"
        ),
    )
    batch.add_argument("--responses-dir", help="Directory with per-sample .txt files: {dir}/{task}/{id}.txt")
    batch.add_argument("--samples", type=int, default=5, help="Samples per task (default: 5)")
    batch.add_argument(
        "--calibrate-n", type=int, default=0, metavar="N",
        help=(
            "Show N gold anchor examples to the judge before each evaluation. "
            "Improves alignment on tasks where the judge misunderstands task type "
            "(e.g. MedCollab, MedDBOps). Recommended: 2. Default: 0 (disabled)."
        ),
    )
    batch.add_argument(
        "--calibrate-mode",
        choices=["random", "bm25", "embedding"],
        default="random",
        help=(
            "How to select calibration anchor examples (requires --calibrate-n > 0). "
            "'random' (default): fixed-seed random sample per task. "
            "'bm25': BM25 semantic retrieval per question (zero extra dependencies). "
            "'embedding': SiliconFlow BAAI/bge-m3 embeddings per question "
            "(requires MINIMAX_API_KEY and MINIMAX_BASE_URL env vars)."
        ),
    )
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

    # generate command (answer model → JSONL for batch)
    from judge.generate_dut import add_generate_arguments

    gen = subparsers.add_parser(
        "generate",
        help="Generate DUT answers with an LLM (default DeepSeek); write JSONL for batch",
    )
    add_generate_arguments(gen)

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
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
