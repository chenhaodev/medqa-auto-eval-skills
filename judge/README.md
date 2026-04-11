# `judge/` package (four modules)

Keep imports stable: **`refs`** (paths + protocol data), **`llm_client`** (remote judge APIs), **`scoring`** (single-sample judge), **`runner`** (batch loop, DUT file I/O, gold RAG, reports).

| Module | Role |
|--------|------|
| `refs.py` | Repo paths; `references/capabilities.json` + `references/rubrics.yaml`; `python -m judge.refs` regenerates `references/rubrics.md`. |
| `llm_client.py` | Judge backends: Anthropic, MiniMax, DeepSeek. |
| `scoring.py` | `judge_response` / `judge_against_gold` and prompt assembly. |
| `runner.py` | DUT response file parsing, `GoldAnchorIndex` (BM25/embedding), `run_benchmark`, `save_results` / `print_summary`. |

**Dependency sketch:** `runner` → `scoring` → `llm_client`; `runner` → `refs`; `scoring` → `refs`.
