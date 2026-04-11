"""
Everything loaded from ``references/``: repo paths, capability groups, rubrics.

Regenerate ``references/rubrics.md`` after editing ``rubrics.yaml``:

    python -m judge.refs
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Paths (repo root + default benchmark dir) ───────────────────────────────

DEFAULT_BENCHMARK_REL = "references/medbench-agent-95"


def repo_root() -> Path:
    """Repository root directory."""
    return _REPO_ROOT


def default_benchmark_dir() -> Path:
    """Absolute path to shipped references/medbench-agent-95."""
    return (_REPO_ROOT / DEFAULT_BENCHMARK_REL).resolve()


def resolve_benchmark_dir(path: str | Path | None) -> str:
    """
    Resolve a benchmark directory to an absolute path string.

    Relative paths are interpreted from the repository root (not cwd).
    """
    if path is None or str(path).strip() == "":
        return str(default_benchmark_dir())
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    cand = (_REPO_ROOT / p).resolve()
    if cand.exists():
        return str(cand)
    parts = p.parts
    if parts == ("medbench-agent-95",) or parts == (".", "medbench-agent-95"):
        alt = default_benchmark_dir()
        if alt.exists():
            return str(alt)
    return str(cand)


# ── Capabilities (references/capabilities.json) ───────────────────────────────

_CAP_JSON = _REPO_ROOT / "references" / "capabilities.json"


@dataclass(frozen=True)
class CapabilityGroup:
    name: str
    key: str
    description: str
    tasks: tuple[str, ...]
    example_question: str


def _load_capability_groups() -> tuple[dict[str, CapabilityGroup], tuple[str, ...]]:
    raw = json.loads(_CAP_JSON.read_text(encoding="utf-8"))
    order: list[str] = []
    groups: dict[str, CapabilityGroup] = {}
    for item in raw["groups"]:
        key = item["key"]
        order.append(key)
        groups[key] = CapabilityGroup(
            key=key,
            name=item["name"],
            description=item["description"],
            tasks=tuple(item["tasks"]),
            example_question=item.get("example_question", ""),
        )
    return groups, tuple(order)


CAPABILITY_GROUPS, _CAPABILITY_ORDER = _load_capability_groups()


def get_tasks_for_capability(capability_key: str) -> tuple[str, ...]:
    """Return task list for a capability key. Raises ValueError if unknown."""
    if capability_key not in CAPABILITY_GROUPS:
        available = ", ".join(CAPABILITY_GROUPS.keys())
        raise ValueError(
            f"Unknown capability '{capability_key}'. Available: {available}"
        )
    return CAPABILITY_GROUPS[capability_key].tasks


def list_capabilities() -> list[CapabilityGroup]:
    """Return capability groups in display order."""
    return [CAPABILITY_GROUPS[k] for k in _CAPABILITY_ORDER if k in CAPABILITY_GROUPS]


# ── Rubrics (references/rubrics.yaml) ───────────────────────────────────────

_RUBRICS_YAML = _REPO_ROOT / "references" / "rubrics.yaml"


@dataclass(frozen=True)
class Criterion:
    name: str
    description: str
    score_1: str
    score_5: str


@dataclass(frozen=True)
class Rubric:
    task: str
    description: str
    criteria: List[Criterion]
    uses_gold_answer: bool = True


def _load_rubrics() -> tuple[dict[str, Rubric], tuple[str, ...], dict[str, str]]:
    raw = yaml.safe_load(_RUBRICS_YAML.read_text(encoding="utf-8"))
    doc_order = tuple(raw["doc_order"])
    subtitles: dict[str, str] = dict(raw["subtitles"])
    rubrics: dict[str, Rubric] = {}
    for key, rdata in raw["rubrics"].items():
        crits = [
            Criterion(
                name=c["name"],
                description=c["description"],
                score_1=c["score_1"],
                score_5=c["score_5"],
            )
            for c in rdata["criteria"]
        ]
        rubrics[key] = Rubric(
            task=rdata["task"],
            description=rdata["description"],
            criteria=crits,
            uses_gold_answer=rdata.get("uses_gold_answer", True),
        )
    return rubrics, doc_order, subtitles


RUBRICS, RUBRIC_DOC_ORDER, RUBRIC_SUBTITLE = _load_rubrics()


def get_rubric(task: str) -> Rubric:
    """Return rubric for the given task name."""
    if task not in RUBRICS:
        available = ", ".join(RUBRICS.keys())
        raise ValueError(f"Unknown task '{task}'. Available: {available}")
    return RUBRICS[task]


def list_tasks() -> list[str]:
    """Return sorted list of all supported task names."""
    return sorted(RUBRICS.keys())


def _esc_rubric_cell(cell: str) -> str:
    return " ".join(cell.split()).replace("|", "\\|")


def _render_rubric_doc(rubric: Rubric) -> str:
    lines: list[str] = []
    task = rubric.task
    sub = RUBRIC_SUBTITLE.get(task, task)
    lines.append(f"## {task} — {sub}")
    lines.append("")
    lines.append(f"*{rubric.description.strip()}*")
    lines.append("")
    lines.append("| Criterion | Description | Score 1 | Score 5 |")
    lines.append("|-----------|-------------|---------|---------|")
    for c in rubric.criteria:
        row = (
            f"| {_esc_rubric_cell(c.name)} | {_esc_rubric_cell(c.description)} | "
            f"{_esc_rubric_cell(c.score_1)} | {_esc_rubric_cell(c.score_5)} |"
        )
        lines.append(row)
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def write_rubrics_markdown(out_path: Path | None = None) -> Path:
    """Write references/rubrics.md from loaded RUBRICS."""
    missing = [t for t in RUBRIC_DOC_ORDER if t not in RUBRICS]
    if missing:
        raise ValueError(f"RUBRICS missing tasks: {missing}")
    extra = set(RUBRICS.keys()) - set(RUBRIC_DOC_ORDER)
    if extra:
        raise ValueError(
            f"RUBRICS has tasks not in RUBRIC_DOC_ORDER: {sorted(extra)}"
        )

    out = out_path or (_REPO_ROOT / "references" / "rubrics.md")
    parts: list[str] = [
        "# MedBench-Eval Task Rubrics",
        "",
        "Auto-generated by `python -m judge.refs` from [`rubrics.yaml`](rubrics.yaml). "
        "Do not edit by hand — change the YAML and re-run.",
        "",
        "Each criterion is scored 1 (worst) to 5 (best) on a Likert scale.",
        "",
        "---",
        "",
    ]
    for task in RUBRIC_DOC_ORDER:
        parts.append(_render_rubric_doc(RUBRICS[task]))

    parts.extend(
        [
            "## Scoring Reference",
            "",
            "| Normalized Score | Interpretation |",
            "|-----------------|----------------|",
            "| 90-100 | Exceptional — meets or exceeds gold standard |",
            "| 75-89 | Strong — minor gaps only |",
            "| 60-74 | Adequate — some important gaps |",
            "| 40-59 | Weak — significant deficiencies |",
            "| 0-39 | Poor — major failures |",
            "",
            "**Formula:** `normalized = (average criterion score − 1) / 4 × 100`, "
            "where each criterion is 1–5.",
            "",
        ]
    )

    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    try:
        write_rubrics_markdown()
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        raise SystemExit(1) from err
