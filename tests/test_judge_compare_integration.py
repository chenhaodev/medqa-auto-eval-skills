"""
Optional LLM judge integration: validate.py Mode 2 (gold vs real DUT).

- Does **not** use Claude Code / OpenCode or SKILL.md — plain subprocess + API key.
- **Skipped by default** to avoid cost; enable with:

      RUN_JUDGE_COMPARE=1 ANTHROPIC_API_KEY=... \\
        uv run --group dev pytest tests/test_judge_compare_integration.py -v

- Uses `tests/fixtures/gpt41_agent_slice` (checked-in GPT-4.1 answers) unless
  `MEDBENCH_COMPARE_DIR` points at a full `MedBench_Agent` tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_AGENT = REPO_ROOT / "tests/fixtures/gpt41_agent_slice"
DEFAULT_USER_AGENT = (
    Path.home()
    / "MedBench/medbench-env/MedBench-2026/SUBMIT/"
    "results-gpt-4.1-20251216/MedBench_Agent"
)


def _compare_dir() -> Path:
    env = os.environ.get("MEDBENCH_COMPARE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if DEFAULT_USER_AGENT.is_dir():
        return DEFAULT_USER_AGENT.resolve()
    return FIXTURE_AGENT


def _should_run_llm_compare() -> bool:
    return (
        os.environ.get("ANTHROPIC_API_KEY", "").strip() != ""
        and os.environ.get("RUN_JUDGE_COMPARE", "").lower() in ("1", "true", "yes")
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _should_run_llm_compare(),
    reason="Set RUN_JUDGE_COMPARE=1 and ANTHROPIC_API_KEY to run (calls Claude API).",
)
def test_validate_mode2_medcot_gold_vs_gpt41_gap() -> None:
    """
    Mode 2: same questions as references/medbench-agent-95, DUT = GPT-4.1 export.

    Human reference: gold ~95 vs DUT ~81 overall — judge should tend to score
    gold above DUT on matched ids (discrimination; gap direction).
    """
    compare_dir = _compare_dir()
    assert (compare_dir / "MedCOT.jsonl").is_file(), f"missing MedCOT in {compare_dir}"

    with tempfile.TemporaryDirectory() as tmp:
        out_base = Path(tmp)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.validate",
                "--task",
                "MedCOT",
                "--samples",
                "2",
                "--compare-dir",
                str(compare_dir),
                "--compare-name",
                "gpt-4.1",
                "--compare-expected-score",
                "81",
                "--model",
                "claude-haiku-4-5",
                "--delay",
                "0",
                "--output-dir",
                str(out_base),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = proc.stdout + proc.stderr
    assert "MedCOT" in out
    assert "gap=" in out
