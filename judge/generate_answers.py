"""
Generate benchmark answers with an LLM (e.g. DeepSeek), then score with ``eval.py batch``.

Uses the same sample selection as ``run_benchmark`` (seeded random subset) so
``--samples`` / ``--seed`` align with a follow-up ``eval.py batch`` call.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

from .llm_client import DEEPSEEK_CHAT, call_model
from .refs import default_benchmark_dir, get_tasks_for_capability, resolve_benchmark_dir
from .runner import BENCHMARK_TASKS, load_task_jsonl_samples

ANSWER_SYSTEM_BASE = (
    "You are a medical AI assistant. Follow the user's instructions and any "
    "referenced evaluation protocol carefully. Answer in the same language as "
    "the question unless asked otherwise."
)


def strip_yaml_frontmatter(markdown_text: str) -> str:
    """Remove leading ``--- ... ---`` YAML block if present."""
    lines = markdown_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return markdown_text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1 :]).lstrip()
    return markdown_text


def build_answer_system(skill_text: Optional[str]) -> str:
    if not skill_text or not skill_text.strip():
        return ANSWER_SYSTEM_BASE
    body = strip_yaml_frontmatter(skill_text.strip())
    return f"{ANSWER_SYSTEM_BASE}\n\n---\n\n{body}"


def resolve_tasks(
    benchmark_path: Path,
    task: Optional[str],
    capability: Optional[str],
) -> list[str]:
    if capability:
        tasks = list(get_tasks_for_capability(capability))
    elif task:
        tasks = [task]
    else:
        tasks = list(BENCHMARK_TASKS)
    return [t for t in tasks if (benchmark_path / f"{t}.jsonl").is_file()]


def generate_jsonl_for_task(
    *,
    benchmark_dir: str,
    task: str,
    answer_model: str,
    system_prompt: str,
    samples_per_task: int,
    seed: int,
    delay_seconds: float,
    verbose: bool,
) -> list[dict]:
    """Return one JSON object per line (same schema as gold JSONL, ``answer`` = model)."""
    bench = Path(benchmark_dir)
    samples = load_task_jsonl_samples(bench, task)
    if not samples:
        return []

    rng = random.Random(seed)
    selected = rng.sample(samples, min(samples_per_task, len(samples)))
    out: list[dict] = []

    for i, sample in enumerate(selected):
        question = str(sample.get("question", ""))
        other = sample.get("other", {})
        if not isinstance(other, dict):
            other = {}
        sample_id = other.get("id", i)

        if verbose:
            print(f"  [{i + 1}/{len(selected)}] {task} id={sample_id} ...", flush=True)

        try:
            resp = call_model(question, model=answer_model, system=system_prompt)
            answer_text = resp.content
        except Exception as e:
            answer_text = f"[GENERATION_FAILED: {e}]"
            print(f"    ERROR id={sample_id}: {e}", file=sys.stderr, flush=True)

        row = {
            "question": question,
            "answer": answer_text,
            "other": {**other, "id": sample_id},
        }
        out.append(row)

        if delay_seconds > 0 and i < len(selected) - 1:
            time.sleep(delay_seconds)

    return out


def run_generate_answers(args: argparse.Namespace) -> None:
    benchmark_path = Path(resolve_benchmark_dir(args.benchmark))
    if not benchmark_path.is_dir():
        print(f"ERROR: benchmark dir not found: {benchmark_path}", file=sys.stderr)
        sys.exit(1)

    skill_text: Optional[str] = None
    if getattr(args, "no_skill", False):
        skill_text = None
    elif args.skill_path:
        p = Path(args.skill_path).expanduser()
        if not p.is_file():
            print(f"ERROR: --skill-path not a file: {p}", file=sys.stderr)
            sys.exit(1)
        skill_text = p.read_text(encoding="utf-8")
    else:
        default_skill = Path(__file__).resolve().parent.parent / "SKILL.md"
        if default_skill.is_file():
            skill_text = default_skill.read_text(encoding="utf-8")

    system_prompt = build_answer_system(skill_text)

    tasks = resolve_tasks(benchmark_path, args.task, args.capability)
    if not tasks:
        print("ERROR: no matching task JSONL files in benchmark dir.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        rows = generate_jsonl_for_task(
            benchmark_dir=str(benchmark_path),
            task=task,
            answer_model=args.answer_model,
            system_prompt=system_prompt,
            samples_per_task=args.samples,
            seed=args.seed,
            delay_seconds=args.delay,
            verbose=not args.quiet,
        )
        if not rows:
            print(f"[{task}] skip: no samples", file=sys.stderr)
            continue
        dest = out_dir / f"{task}.jsonl"
        with open(dest, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not args.quiet:
            print(f"Wrote {len(rows)} rows -> {dest}")

    if not args.quiet:
        print(
            "\nNext: score with the judge (same --samples / --seed as here), e.g.\n"
            f"  python eval.py batch --task ... --dut {args.dut_label} "
            f"--responses-file {out_dir}/MedCOT.jsonl "
            f"--samples {args.samples} --seed {args.seed} --output results/run/",
        )


def add_generate_answers_arguments(p: argparse.ArgumentParser) -> None:
    """CLI flags for ``eval.py generate`` and ``python -m judge.generate_answers``."""
    p.add_argument(
        "--benchmark",
        default=str(default_benchmark_dir()),
        help="Gold benchmark directory (default: shipped medbench-agent-95)",
    )
    p.add_argument(
        "--task",
        help="Single task (e.g. MedCOT). Overrides --capability when set.",
    )
    p.add_argument(
        "--capability",
        choices=[
            "reasoning",
            "long_context",
            "tool_use",
            "orchestration",
            "self_correction",
            "role_adapt",
            "safety",
            "full",
        ],
        help="Task group (used when --task is omitted)",
    )
    p.add_argument(
        "--answer-model",
        default=DEEPSEEK_CHAT,
        help=(
            "Model id for answers: deepseek-chat, deepseek-reasoner, "
            "minimax-m2.5, or claude-* (see judge/llm_client.py). "
            f"Default: {DEEPSEEK_CHAT}"
        ),
    )
    p.add_argument(
        "--skill-path",
        metavar="PATH",
        help="Markdown file to inject as system context after the base instruction "
        "(default: repo SKILL.md if present).",
    )
    p.add_argument(
        "--no-skill",
        action="store_true",
        help="Do not load SKILL.md or --skill-path; base system prompt only.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Samples per task (same RNG rule as eval.py batch; default: 5)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between answer API calls (default: 0.5)",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to write {Task}.jsonl files",
    )
    p.add_argument(
        "--dut-label",
        default="deepseek-chat",
        help="Printed hint only; use the same string in eval.py batch --dut",
    )
    p.add_argument("--quiet", action="store_true")


def build_generate_answers_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate benchmark answers with an answer model (default: DeepSeek), "
            "optionally injecting SKILL.md as system context. "
            "Writes per-task JSONL compatible with eval.py batch --responses-file."
        ),
    )
    add_generate_answers_arguments(p)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_generate_answers_parser()
    args = parser.parse_args(argv)
    run_generate_answers(args)


if __name__ == "__main__":
    main()
