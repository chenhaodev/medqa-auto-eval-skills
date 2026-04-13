# Protocol vs code

This repository mixes **human-editable specifications** with **Python that must run** (API calls, batch loops, parsing). The goal is to keep **static policy** out of `.py` where possible so TUI / skill workflows do not depend on “reading the codebase.”

| Artifact | Role |
|----------|------|
| [`SKILL.md`](../SKILL.md) | In-chat judging protocol for `/medbench-eval` (no Python required). |
| [`capabilities.json`](capabilities.json) | Capability groups → task lists for the wizard and `eval.py batch --capability`. Edit this JSON to change grouping; `judge/refs.py` loads it. |
| [`rubrics.yaml`](rubrics.yaml) | **Source** for task rubrics and criteria (Likert anchors). |
| [`rubrics.md`](rubrics.md) | Human-readable copy for models (generated: `python -m judge.refs`). |
| [`medbench-agent-95/`](medbench-agent-95/) | Gold questions/answers (JSONL). Data only. |

**Why Python remains:** calling judge LLMs (`judge/llm_client.py`), scoring prompts (`judge/scoring.py`), and the combined batch pipeline (`judge/runner.py`: benchmark walk, DUT file I/O, gold RAG, reports), plus optional alignment (`scripts/validate.py`), are **not** replaceable by markdown alone without losing automation.

**History:** rubric criteria used to live only in Python; they now live in **`references/rubrics.yaml`** and are loaded by **`judge/refs.py`**. Regenerate `rubrics.md` with `python -m judge.refs`.

---

## Sampling algorithm (seed-42)

Knowing this exactly lets agents reproduce the same question IDs as `eval.py batch`.

**Python batch (`run_benchmark`):**

```python
rng = random.Random(seed)          # one RNG created before the task loop; default seed=42
for task in tasks_to_run:          # tasks in deterministic list order
    rows = [json.loads(l) for l in open(f"{task}.jsonl")]  # file order, no pre-shuffle
    selected = rng.sample(rows, min(N, len(rows)))
```

The RNG is **shared across tasks**, so which rows get selected for task K depends on how many were drawn from tasks 1…K-1. A single-task run gives different IDs than the same task inside a full capability batch.

**In-chat wizard approximation (no Python):**

Seed **per task independently** — treat each task as if it runs first:

```
selected_ids = Random(42).sample(all_row_ids_in_file_order, N)
```

This is close enough for in-chat use (and fully reproducible) even though it diverges from batch output for tasks after the first.
