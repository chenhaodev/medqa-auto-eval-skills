# `judge/` package map

| Module | Role |
|--------|------|
| `scoring.py` | Single-call judge: score one DUT reply (`judge_response` / `judge_against_gold`). |
| `batch_runner.py` | Walk gold JSONL, sample, optional calibration anchors, aggregate `BenchmarkResult`. |
| `llm_client.py` | Judge backends: Anthropic Haiku, MiniMax, DeepSeek, etc. |
| `dut_responses.py` | Parse DUT outputs from dirs, JSONL, or delimited text. |
| `gold_retrieval.py` | `GoldAnchorIndex`: BM25 or embedding retrieval over gold for `--calibrate-n`. |
| `rubrics.py` | 13 task rubrics (source of truth; `references/rubrics.md` is generated). |
| `capabilities.py` | Capability groups → task lists (wizard / `--capability`). |
| `report.py` | Write `summary.json`, `details.jsonl`, `report.md`. |
| `paths.py` | Default benchmark path `references/medbench-agent-95`. |

**Dependency sketch:** `batch_runner` → `scoring` → `llm_client`; `batch_runner` → `gold_retrieval`, `dut_responses`.
