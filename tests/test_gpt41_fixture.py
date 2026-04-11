"""
Offline checks for the checked-in GPT-4.1 MedBench_Agent slice (DUT JSONL).

Does not call any judge API. Does not load SKILL — only file alignment with gold.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests/fixtures/gpt41_agent_slice"
GOLD_MEDCOT = REPO_ROOT / "references/medbench-agent-95/MedCOT.jsonl"


def test_fixture_medcot_exists() -> None:
    p = FIXTURE_DIR / "MedCOT.jsonl"
    assert p.is_file()
    assert p.stat().st_size > 100


def test_fixture_ids_match_gold_for_first_rows() -> None:
    """DUT lines must share sample ids with shipped gold for validate Mode 2 pairing."""
    gold_ids: list[int] = []
    with open(GOLD_MEDCOT, encoding="utf-8") as f:
        for _ in range(3):
            line = next(l for l in f if l.strip())
            rec = json.loads(line)
            gold_ids.append(int(rec["other"]["id"]))

    dut_ids: list[int] = []
    with open(FIXTURE_DIR / "MedCOT.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            dut_ids.append(int(rec["other"]["id"]))

    assert dut_ids == gold_ids[: len(dut_ids)]


def test_fixture_parse_responses_via_runner() -> None:
    from judge.runner import parse_responses

    got = parse_responses(FIXTURE_DIR / "MedCOT.jsonl")
    assert "MedCOT" in got
    assert len(got["MedCOT"]) >= 1
