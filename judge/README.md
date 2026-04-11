# `judge/` package

Keep imports stable: **`refs`** (paths + protocol data), **`llm_client`** (remote judge APIs), **`scoring`** (single-sample judge), **`runner`** (batch loop, DUT file I/O, gold RAG, reports), **`generate_answers`** (optional answer generation for `eval.py generate`).

| Module | Role |
|--------|------|
| `refs.py` | Repo paths; `references/capabilities.json` + `references/rubrics.yaml`; `python -m judge.refs` regenerates `references/rubrics.md`. |
| `llm_client.py` | Judge backends: Anthropic, MiniMax, DeepSeek. |
| `scoring.py` | `judge_response` / `judge_against_gold` and prompt assembly. |
| `runner.py` | DUT response file parsing, `GoldAnchorIndex` (BM25/embedding), `run_benchmark`, `save_results` / `print_summary`. |
| `generate_answers.py` | Sample gold questions, call an answer LLM, write JSONL for `eval.py batch --responses-file`. |

**Dependency sketch:** `runner` → `scoring` → `llm_client`; `runner` → `refs`; `scoring` → `refs`; `generate_answers` → `llm_client`, `runner` (sample loading).
