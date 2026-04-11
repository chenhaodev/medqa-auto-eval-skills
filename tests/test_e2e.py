"""
End-to-end smoke tests: CLIs and offline gold-index paths (no LLM API calls).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(
    argv: list[str],
    *,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_help_exits_zero() -> None:
    proc = _run([sys.executable, "-m", "scripts.validate", "--help"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout + proc.stderr
    assert "validate" in out.lower()


def test_eval_help_exits_zero() -> None:
    proc = _run([sys.executable, "eval.py", "--help"])
    assert proc.returncode == 0, proc.stderr
    assert "medbench-eval" in (proc.stdout + proc.stderr).lower()


def test_eval_tasks_lists_medcot() -> None:
    proc = _run([sys.executable, "eval.py", "tasks"])
    assert proc.returncode == 0, proc.stderr
    assert "MedCOT" in proc.stdout


def test_capabilities_protocol_json() -> None:
    """references/capabilities.json drives judge.refs (8 groups)."""
    from judge.refs import CAPABILITY_GROUPS, list_capabilities

    assert len(CAPABILITY_GROUPS) == 8
    assert len(list_capabilities()) == 8


def test_rubric_doc_order_matches_rubrics() -> None:
    """Ensures RUBRIC_DOC_ORDER matches RUBRICS keys (export invariant)."""
    from judge.refs import RUBRIC_DOC_ORDER, RUBRICS

    missing = [t for t in RUBRIC_DOC_ORDER if t not in RUBRICS]
    assert not missing
    extra = set(RUBRICS.keys()) - set(RUBRIC_DOC_ORDER)
    assert not extra


def test_rubrics_yaml_on_disk() -> None:
    """Protocol file exists next to repo usage."""
    yaml_path = REPO_ROOT / "references" / "rubrics.yaml"
    assert yaml_path.is_file()
    assert yaml_path.stat().st_size > 1000


def test_shipped_benchmark_has_thirteen_jsonl() -> None:
    from judge.refs import default_benchmark_dir

    bd = default_benchmark_dir()
    assert bd.is_dir(), f"missing benchmark dir: {bd}"
    jsonl = sorted(bd.glob("*.jsonl"))
    assert len(jsonl) == 13, f"expected 13 *.jsonl, got {len(jsonl)} in {bd}"


def test_gold_anchor_index_bm25_retrieve() -> None:
    from judge.refs import default_benchmark_dir
    from judge.runner import GoldAnchorIndex

    idx = GoldAnchorIndex(default_benchmark_dir(), task="MedCOT", backend="bm25")
    assert len(idx) >= 1
    got = idx.retrieve(question="高血压", n=2, exclude_ids=set())
    assert len(got) >= 1
    assert "question" in got[0] and "answer" in got[0]


@pytest.mark.parametrize(
    "task_file",
    [
        "MedCOT.jsonl",
        "MedShield.jsonl",
    ],
)
def test_sample_benchmark_lines_are_valid_json(task_file: str) -> None:
    import json

    from judge.refs import default_benchmark_dir

    path = default_benchmark_dir() / task_file
    assert path.is_file()
    with open(path, encoding="utf-8") as f:
        first = next(line for line in f if line.strip())
    data = json.loads(first)
    assert "question" in data and "answer" in data
