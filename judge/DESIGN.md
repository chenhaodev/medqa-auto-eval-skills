# judge/ — Design Notes

Plain-language explanation of each module for agents and contributors who do not want to read Python.

## Module map

```
judge/
  refs.py             ← load references/; capability + rubric lookups
  llm_client.py       ← call any judge LLM (Claude / MiniMax / DeepSeek)
  scoring.py          ← build judge prompt, parse JSON response, compute score
  runner.py           ← DUT file I/O · gold index · benchmark loop · reports
  generate_answers.py ← generate DUT answers and write JSONL for batch
```

---

## refs.py — reference data loader

Reads three files from `references/` at import time and exposes them as typed Python objects:

| Source file | What it provides |
|-------------|-----------------|
| `capabilities.json` | `CAPABILITY_GROUPS`: maps `key` (e.g. `reasoning`) → list of task names |
| `rubrics.yaml` | `RUBRICS`: maps task name → `Rubric(description, criteria[])` |
| (derived) | `list_tasks()` — sorted list of all 13 task names |

Key functions:
- `get_rubric(task)` — returns the `Rubric` dataclass for a task
- `get_tasks_for_capability(key)` — returns task tuple for a capability key
- `write_rubrics_markdown()` — regenerates `references/rubrics.md` from YAML; run with `python -m judge.refs`

All paths are resolved relative to the repo root (`Path(__file__).parent.parent`), so `python -m judge.refs` works from any cwd.

---

## llm_client.py — model abstraction

Single entry point: `call_model(prompt, model, system) → ModelResponse`.

Routes by model identifier prefix:
- `claude-*` → Anthropic SDK (`ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`)
- `minimax-m2.5` / `minimax` → OpenAI-compatible via SiliconFlow (`MINIMAX_API_KEY`, `MINIMAX_BASE_URL`)
- `deepseek-chat` / `deepseek-reasoner` → DeepSeek API (`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`)

Default judge model: `claude-haiku-4-5` (`DEFAULT_MODEL`). All non-Claude backends use `temperature=0.0` and `max_tokens=2048`. No external HTTP library — uses stdlib `urllib.request`.

Loads `.env` from repo root automatically on import (no `python-dotenv` required).

---

## scoring.py — judge logic

**What it does in plain English:**

1. Look up the rubric for the task (`refs.get_rubric`)
2. Build a prompt that shows the judge: task description, optional score-5 calibration examples, the question, optional gold answer, the DUT response, and each criterion with score-1 / score-5 anchors
3. Instruct the judge to return **JSON only** — no prose — in the shape:
   ```json
   {
     "criterion_scores": { "<criterion>": { "score": 4, "justification": "..." } },
     "overall_feedback": "...",
     "minor_errors_noted": [...],
     "major_errors_noted": [...]
   }
   ```
4. Parse the JSON; clamp each score to `[1, 5]`
5. Compute: `total = mean(criterion scores)` → `normalized = (total − 1) / 4 × 100`

**Scoring philosophy baked into the system prompt:**
- Penalize wrong clinical facts, unsafe omissions, bad reasoning
- Do NOT penalize synonym phrasing, formatting style, extra correct detail, reordered steps

Two public functions:
- `judge_response(task, question, response, gold_answer, ...)` — evaluate a DUT response
- `judge_against_gold(task, question, gold_answer, ...)` — evaluate gold against itself (ceiling check)

---

## runner.py — benchmark pipeline

Four separate concerns in one file:

### 1. DUT file I/O (`parse_responses`, `parse_responses_text`)

Auto-detects format and returns `ParsedResponses = dict[task → dict[id → response_text]]`.

Formats tried in order:
1. **Directory** — scans `*.jsonl` files, reads `other.id` + `answer` fields
2. **JSONL** — lines with `{task, id, response}` or `{id, response}` or `{other: {source, id}, answer}`
3. **Delimited text** — headers matching `=== TaskName 97 ===` (also `---`, `##` variants)
4. **Sequential** — blank-line-separated blocks or `===` separators; IDs assigned by position

### 2. Gold anchor index (`GoldAnchorIndex`)

Optional few-shot calibration: retrieves similar gold examples to show the judge before scoring.

Two backends:
- `bm25` (default, no API) — custom BM25 implementation; supports CJK character tokenization
- `embedding` — calls SiliconFlow `BAAI/bge-m3` via `MINIMAX_API_KEY`

Used when `--calibrate-n N` is passed to `eval.py batch`.

### 3. Benchmark loop (`run_benchmark`)

Helpers: `select_rows_for_task(rng, rows, N)` and
`sample_rows_shared_rng_for_tasks(benchmark_dir, tasks, N, seed)` — pure
selection for tests and parity checks (no LLM).

```
for each task:
    load all JSONL rows from benchmark_dir/{task}.jsonl
    select N rows with shared RNG(seed=42)      ← see sampling note below
    for each selected row:
        fetch DUT response (from file / responses_file / gold fallback)
        optionally retrieve calibration anchors
        call scoring.judge_response or judge_against_gold
        collect SampleResult
```

Returns `BenchmarkResult` — a dict of task → list of `SampleResult`.

### 4. Reports (`build_summary`, `save_results`)

`save_results(result, output_dir)` writes three files:
- `summary.json` — metadata + per-task avg/min/max + criterion means
- `details.jsonl` — one line per sample with full criterion scores and error lists
- `report.md` — human-readable markdown table

---

## Sampling algorithm (exact specification)

This matters for reproducibility between `eval.py batch` and the in-chat wizard.

**Batch mode (`run_benchmark`):**

```python
rng = random.Random(seed)          # created ONCE before the task loop
for task in tasks_to_run:          # tasks in deterministic order
    samples = load_jsonl(task)     # rows in file order (no shuffle before sampling)
    selected = rng.sample(samples, min(N, len(samples)))
```

The RNG is **shared across tasks** — samples for task K depend on how many were drawn for tasks 1…K-1. Running a single task gives different IDs than running it as part of a full capability batch.

**In-chat wizard approximation:**

When an agent exports questions without running Python, use:
- Seed the RNG at 42 **per task independently** (i.e. treat each task as if it runs first)
- `random.Random(42).sample(all_rows_in_file_order, N)`
- Row order = JSONL file line order

This will not exactly match `eval.py batch` output for tasks beyond the first, but is close enough for in-chat use and is fully reproducible across sessions.

**`eval.py generate`:** uses the **same shared RNG** as `run_benchmark` (one `Random(seed)` across tasks in order), so multi-task answer generation matches batch sampling for the same `--samples` / `--seed`.
