# CLAUDE.md

Guidance for **Claude Code** (claude.ai/code) in this repository. User-facing install and CLI examples live in **[`README.md`](README.md)**; in-chat `/medbench-eval` behavior lives in **[`SKILL.md`](SKILL.md)**.

## What this repo does

LLM-as-judge on **MedBench-Agent-95** (390 items, 13 task types). Likert 1–5 per criterion → normalized 0–100 = `(avg − 1) / 4 × 100`. Default judge: `claude-haiku-4-5`; default answer model for `eval.py generate`: `deepseek-chat`.

**Entry points:** `eval.py` (CLI + wizard with no args) · `SKILL.md` (skill protocol, no Python).

## Setup

```bash
uv sync --group dev
cp .env.example .env   # ANTHROPIC_API_KEY required for judge; DEEPSEEK_API_KEY for eval.py generate
```

Run commands from the **repository root** (`judge/refs.py` resolves `references/medbench-agent-95/` relative to cwd).

## Commands (quick)

```bash
uv run --group dev pytest tests/ -q          # default: no API calls
uv run --group dev pytest tests/ -m integration   # optional: real API
python eval.py tasks
python eval.py batch --capability reasoning --dut MY_MODEL --samples 5 --output results/run-001/
python -m judge.refs                       # after editing references/rubrics.yaml
```

More: batch/generate/validate → **README** § Run evaluations & Judge alignment.

## Architecture

```
eval.py
judge/
  refs.py             # paths; capabilities.json + rubrics.yaml; regenerate rubrics.md
  llm_client.py       # Anthropic, MiniMax, DeepSeek
  scoring.py          # judge_response / judge_against_gold
  runner.py           # parse DUT files, run_benchmark, reports
  generate_answers.py # answers → JSONL for batch
scripts/validate.py
references/
  capabilities.json   # capability → tasks (edit JSON)
  rubrics.yaml        # source rubrics (edit here)
  rubrics.md          # generated — do not edit by hand
  medbench-agent-95/  # gold JSONL per task
```

**Flow:** `runner` → `scoring` → `llm_client`; `runner` → `refs`; `generate_answers` → `llm_client`, `runner`.

## Rules

- **Policy in data:** rubrics in `references/rubrics.yaml`; groupings in `references/capabilities.json`. Do not hardcode criteria or task lists in `.py`.
- **Regenerate** `references/rubrics.md` after `rubrics.yaml` edits: `python -m judge.refs`.
- **Dependencies:** change `pyproject.toml` → `uv lock` → `uv export --format requirements-txt -o requirements.txt` → commit all three together.
- **Integration tests** (`@pytest.mark.integration`) are opt-in and call external APIs.
