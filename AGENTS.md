# AGENTS.md

LLM-as-judge evaluation framework for MedBench-Agent-95 (390 items, 13 task types). Scores are Likert 1–5 per criterion → normalized 0–100.

## Two modes of operation

**Interactive (no API keys needed):** The OpenCode agent IS the judge. Read rubrics from `references/rubrics.md`, export questions for the user to test on their DUT (DeepSeek, etc.), accept answers back, score in-chat. This is the primary `/medbench-eval` skill flow.

**CLI automation (requires API keys):** `eval.py batch` / `eval.py generate` / `scripts.validate` call external LLM APIs for judging or answer generation. Only needed for unattended batch runs.

## Setup

```bash
uv sync --group dev
# .env only needed for CLI automation (ANTHROPIC_API_KEY, DEEPSEEK_API_KEY)
```

**Always run commands from the repo root** — `judge/refs.py` resolves `references/medbench-agent-95/` relative to the repo root via `Path(__file__).resolve().parent.parent`, not cwd. Running from a subdirectory will break benchmark path resolution.

## Commands

```bash
uv run --group dev pytest tests/ -q              # offline smoke tests (no API calls)
uv run --group dev pytest tests/ -m integration  # real API calls — requires ANTHROPIC_API_KEY
python eval.py tasks                              # list capability groups and tasks
python eval.py batch --capability reasoning --dut MY_MODEL --samples 5 --output results/run-001/
python eval.py single --task MedCOT --question "..." --response "..." --dut NAME
python eval.py generate -o generated/dut/ --task MedCOT --answer-model deepseek-chat
python -m judge.refs                              # regenerate rubrics.md after rubrics.yaml edits
python -m scripts.validate --task MedCOT --samples 3  # judge alignment check (uses API)
```

`eval.py` with no arguments launches an interactive wizard — **unsuitable for unattended agents**.

## Architecture

- `eval.py` — CLI entry point (argparse subcommands: batch, single, generate, tasks)
- `judge/refs.py` — repo paths, loads `capabilities.json` + `rubrics.yaml`; `python -m judge.refs` → regenerates `rubrics.md`
- `judge/llm_client.py` — model abstraction (Anthropic, MiniMax/SiliconFlow, DeepSeek); auto-loads `.env`
- `judge/scoring.py` — prompt assembly + `judge_response` / `judge_against_gold`
- `judge/runner.py` — batch loop, DUT file parsing (JSONL/TXT/dir), `GoldAnchorIndex` (BM25/embedding), reports
- `judge/generate_answers.py` — answer-model pipeline for `eval.py generate`
- `scripts/validate.py` — judge-human alignment: synthetic tier test or real model comparison

**Dependency chain:** `runner` → `scoring` → `llm_client`; `runner` → `refs`; `scoring` → `refs`; `generate_answers` → `llm_client` + `runner`.

## Data policy

- **Rubrics:** edit `references/rubrics.yaml`. `references/rubrics.md` is **generated** — never edit by hand. After YAML changes, run `python -m judge.refs`.
- **Task groupings:** edit `references/capabilities.json`. No Python changes needed for regrouping.
- **Gold benchmark:** `references/medbench-agent-95/{Task}.jsonl` — never modify; these are the source of truth for questions and gold answers.
- **Do not hardcode** criteria names, task lists, or capability groups in `.py` files — all of this comes from the YAML/JSON data files.

## Dependencies

Change `pyproject.toml` → `uv lock` → `uv export --format requirements-txt -o requirements.txt` → commit all three together.

## Testing

- Default `pytest` run is **offline** — no API keys needed.
- `@pytest.mark.integration` tests call external LLM APIs and are opt-in (`-m integration`).
- BM25 index tests hit the shipped JSONL files but not external APIs.
- 13 task JSONL files must exist in `references/medbench-agent-95/` — tests assert this count.

## Environment variables (CLI automation only)

| Variable | Purpose | Required for |
|---|---|---|
| `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` | Judge (default: `claude-haiku-4-5`) | `batch`, `single`, `validate` |
| `DEEPSEEK_API_KEY` | Answer model (default: `deepseek-chat`) | `generate` |
| `MINIMAX_API_KEY` + `MINIMAX_BASE_URL` | MiniMax judge or embedding calibration | `--model minimax-m2.5` or `--calibrate-mode embedding` |

`.env` in repo root is auto-loaded by `judge/llm_client.py`. Do not commit `.env`.

## Model IDs

Judge: `claude-haiku-4-5` (default), `minimax-m2.5`, `deepseek-chat`. Answer: `deepseek-chat` (default), `deepseek-reasoner`.

**In interactive mode, the OpenCode agent IS the judge** — it reads rubrics and scores directly, no model ID or API key needed.

## Calibration options

`eval.py batch --calibrate-n 2 --calibrate-mode bm25` — few-shot gold anchor examples shown to the judge before scoring. Improves alignment on tasks where the judge misjudges task type. `bm25` mode has zero extra deps; `embedding` mode requires SiliconFlow env vars.

## Scoring formula

`normalized = (average criterion score − 1) / 4 × 100` where each criterion is Likert 1–5. Judge scores substance over style — synonym phrasing, markdown differences, and extra correct detail do not lower scores.

## DUT response formats

The runner auto-detects: compact JSONL (`{task, id, response}`), full benchmark JSONL (`{question, answer, other}`), delimited TXT (`=== MedCOT | 42 ===`), per-sample `.txt` files in `{dir}/{task}/{id}.txt`.
