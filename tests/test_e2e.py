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


def test_eval_batch_help_exits_zero() -> None:
    proc = _run([sys.executable, "eval.py", "batch", "--help"])
    assert proc.returncode == 0, proc.stderr
    assert "benchmark" in (proc.stdout + proc.stderr).lower()


def test_eval_single_missing_args_exits_nonzero() -> None:
    proc = _run(
        [
            sys.executable,
            "eval.py",
            "single",
            "--task",
            "MedCOT",
            "--dut",
            "x",
        ]
    )
    assert proc.returncode != 0


def test_resolve_benchmark_dir_none_is_default() -> None:
    from judge.refs import default_benchmark_dir, resolve_benchmark_dir

    assert resolve_benchmark_dir(None) == str(default_benchmark_dir())


def test_resolve_benchmark_dir_empty_string_is_default() -> None:
    from judge.refs import default_benchmark_dir, resolve_benchmark_dir

    assert resolve_benchmark_dir("") == str(default_benchmark_dir())


def test_repo_root_contains_pyproject() -> None:
    from judge.refs import repo_root

    assert (repo_root() / "pyproject.toml").is_file()


def test_list_tasks_has_thirteen() -> None:
    from judge.refs import list_tasks

    assert len(list_tasks()) == 13


def test_get_rubric_medcot_has_five_criteria() -> None:
    from judge.refs import get_rubric

    r = get_rubric("MedCOT")
    assert r.task == "MedCOT"
    assert len(r.criteria) == 5


def test_get_rubric_unknown_raises() -> None:
    from judge.refs import get_rubric

    with pytest.raises(ValueError, match="Unknown task"):
        get_rubric("NotATask")


def test_get_tasks_for_capability_reasoning() -> None:
    from judge.refs import get_tasks_for_capability

    tasks = get_tasks_for_capability("reasoning")
    assert tasks == ("MedCOT", "MedDecomp", "MedPathPlan")


def test_capability_groups_include_expected_keys() -> None:
    from judge.refs import CAPABILITY_GROUPS

    for key in ("reasoning", "full", "safety", "tool_use"):
        assert key in CAPABILITY_GROUPS


def test_rubric_subtitles_cover_doc_order() -> None:
    from judge.refs import RUBRIC_DOC_ORDER, RUBRIC_SUBTITLE

    for task in RUBRIC_DOC_ORDER:
        assert task in RUBRIC_SUBTITLE


def test_write_rubrics_markdown_to_tmp_path(tmp_path: Path) -> None:
    from judge.refs import write_rubrics_markdown

    out = tmp_path / "rubrics-out.md"
    write_rubrics_markdown(out_path=out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# MedBench-Eval Task Rubrics")
    assert "MedCOT" in text


def test_parse_responses_compact_jsonl_file(tmp_path: Path) -> None:
    from judge.runner import parse_responses

    p = tmp_path / "dut.jsonl"
    p.write_text(
        '{"task": "MedCOT", "id": 1, "response": "hello"}\n',
        encoding="utf-8",
    )
    got = parse_responses(p)
    assert got["MedCOT"][1] == "hello"


def test_parse_responses_delimited_txt_file(tmp_path: Path) -> None:
    from judge.runner import parse_responses

    p = tmp_path / "dut.txt"
    p.write_text(
        "=== MedCOT | 42 ===\nline one\nline two\n=== MedShield | 7 ===\nok\n",
        encoding="utf-8",
    )
    got = parse_responses(p)
    assert got["MedCOT"][42].startswith("line one")
    assert got["MedShield"][7].strip() == "ok"


def test_responses_to_jsonl_contains_task_id() -> None:
    from judge.runner import responses_to_jsonl

    blob = responses_to_jsonl({"MedCOT": {9: "abc"}})
    assert '"task": "MedCOT"' in blob
    assert '"id": 9' in blob


def test_gold_anchor_retrieve_respects_max_n() -> None:
    from judge.refs import default_benchmark_dir
    from judge.runner import GoldAnchorIndex

    idx = GoldAnchorIndex(default_benchmark_dir(), task="MedCOT", backend="bm25")
    got = idx.retrieve(question="高血压诊疗", n=2, exclude_ids=set())
    assert len(got) <= 2
    assert all("question" in x and "answer" in x for x in got)


def test_runner_benchmark_tasks_count() -> None:
    from judge.runner import BENCHMARK_TASKS

    assert len(BENCHMARK_TASKS) == 13


def test_llm_client_default_model_string() -> None:
    from judge.llm_client import DEFAULT_MODEL

    assert isinstance(DEFAULT_MODEL, str)
    assert len(DEFAULT_MODEL) > 0


def test_llm_client_available_models_non_empty() -> None:
    from judge.llm_client import available_models

    names = available_models()
    assert isinstance(names, list)
    assert len(names) >= 3
