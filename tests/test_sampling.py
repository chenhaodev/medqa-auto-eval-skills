"""
Unit tests for benchmark row selection (shared RNG vs batch contract).

No API calls.
"""

from __future__ import annotations

import random

import pytest

from judge.refs import default_benchmark_dir, get_tasks_for_capability
from judge.runner import (
    select_rows_for_task,
    sample_rows_shared_rng_for_tasks,
)


def test_select_rows_for_task_empty_samples() -> None:
    rng = random.Random(0)
    assert select_rows_for_task(rng, [], 5) == []


def test_select_rows_for_task_caps_at_len() -> None:
    rng = random.Random(1)
    rows = [{"k": i} for i in range(3)]
    got = select_rows_for_task(rng, rows, 100)
    assert len(got) == 3


def test_sample_rows_shared_rng_reproducible() -> None:
    bd = default_benchmark_dir()
    tasks = list(get_tasks_for_capability("reasoning"))
    a = sample_rows_shared_rng_for_tasks(bd, tasks, 3, 42)
    b = sample_rows_shared_rng_for_tasks(bd, tasks, 3, 42)
    assert _row_ids(a) == _row_ids(b)


def test_shared_rng_second_task_differs_from_isolated_first_task() -> None:
    """MedCOT rows differ when MedDecomp runs first (PROTOCOL.md shared RNG)."""
    bd = default_benchmark_dir()
    alone = sample_rows_shared_rng_for_tasks(bd, ["MedCOT"], 5, 42)
    chain = sample_rows_shared_rng_for_tasks(
        bd, ["MedDecomp", "MedCOT"], 5, 42
    )
    assert _ids_for_task(alone, "MedCOT") != _ids_for_task(chain, "MedCOT")


def test_get_tasks_for_capability_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown capability"):
        get_tasks_for_capability("not_a_real_capability_key")


def _ids_for_task(
    batch: dict[str, list[dict]],
    task: str,
) -> tuple[int, ...]:
    rows = batch.get(task, [])
    out: list[int] = []
    for r in rows:
        other = r.get("other", {})
        if isinstance(other, dict) and "id" in other:
            out.append(int(other["id"]))
    return tuple(out)


def _row_ids(batch: dict[str, list[dict]]) -> dict[str, tuple[int, ...]]:
    return {t: _ids_for_task(batch, t) for t in batch}
