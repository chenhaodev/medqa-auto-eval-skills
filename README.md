# medbench-eval

Use an **LLM judge** to score answers on **MedBench-Agent-95** (390 items, **13 task types**). Each criterion is Likert **1–5**; scores normalize to **0–100**.

| You are… | Start here |
|----------|------------|
| **A person** using Claude / Cursor with the skill | Read **[`SKILL.md`](SKILL.md)** — that is the contract for `/medbench-eval`. This README is for the **repo and CLI**. |
| **An agent** automating setup and commands | Jump to **[Automation checklist](#automation-checklist-for-agents)** (copy-paste blocks below). |
| **A maintainer** changing rubrics, capabilities, or dependencies | See **[Maintainers](#maintainers)**, [`references/PROTOCOL.md`](references/PROTOCOL.md), and [`references/README.md`](references/README.md). |

---

## What you need

1. **Python 3.11+**
2. **Repository root** as the working directory (the folder that contains `pyproject.toml`).
3. **API key** for the default judge: set **`ANTHROPIC_API_KEY`** (model `claude-haiku-4-5`). Other backends need extra env vars — see [`judge/llm_client.py`](judge/llm_client.py).
4. A **`.env`** file in the repo root is loaded automatically. **Do not commit `.env`.**

---

## Install dependencies

**Recommended ([uv](https://docs.astral.sh/uv/))** — reproducible installs from **`uv.lock`**:

```bash
cd /path/to/repo   # must contain pyproject.toml
uv sync            # runtime only; creates .venv/

# include pytest (for tests/README “smoke”)
uv sync --group dev
```

**Using pip only** (no dev tools):

```bash
pip install -r requirements.txt
```

`requirements.txt` is **exported** from the lockfile (`uv export --format requirements-txt -o requirements.txt`). After you edit **`pyproject.toml`** dependencies, run **`uv lock`**, then re-export if pip users need an updated file.

---

## Automation checklist (for agents)

Execute **in order**. Assume the shell’s cwd is the repository root.

| Step | Action | Command or file |
|------|--------|-------------------|
| 1 | Obtain the tree | `git clone <repo-url>` then `cd` into the root, **or** unpack a tarball and `cd` there. |
| 2 | Install **uv** if missing | `curl -LsSf https://astral.sh/uv/install.sh \| sh` — then ensure `uv` is on `PATH` (often `~/.local/bin`). |
| 3 | Install deps + dev | `uv sync --group dev` |
| 4 | Configure secrets | Create `.env` with at least `ANTHROPIC_API_KEY=...` |
| 5 | Verify (no paid API calls in these checks) | `uv run --group dev pytest tests/ -q` then `python -m scripts.validate --help`, `python eval.py tasks`, and `python eval.py generate --help` |
| 6 | Run real judging | Needs a valid key — see **[Run evaluations](#run-evaluations)** |

For conversational scoring conventions (wizard flow, rubric names, output template), read **`SKILL.md`**. This README does not duplicate that protocol.

---

## Run evaluations

**Always run commands from the repository root** so paths like `references/medbench-agent-95` resolve correctly (`judge/refs.py`).

### 1. See what the CLI can do

```bash
python eval.py --help
python eval.py tasks
```

### 2. Batch (typical automation)

```bash
python eval.py batch \
  --capability reasoning \
  --dut YOUR_MODEL_NAME \
  --responses-dir path/to/responses/ \
  --samples 5 \
  --output results/run-001/
```

- **`--benchmark`** defaults to the shipped gold data: `references/medbench-agent-95/`.
- **Outputs** under `--output`: `summary.json`, `details.jsonl`, `report.md`.

### 2b. Generate answers (DeepSeek, etc.) then batch-judge

Use an **answer model** (default `deepseek-chat`; set `DEEPSEEK_API_KEY` in `.env`) to answer gold questions. By default, repo **`SKILL.md`** is injected as **system** context (use `--no-skill` to disable). Sample selection matches **`eval.py batch`** when you use the same **`--samples`** and **`--seed`**.

```bash
python eval.py generate -o generated/dut/ --task MedCOT --samples 3 --seed 42 --answer-model deepseek-chat
python eval.py batch --task MedCOT --dut deepseek-chat \
  --responses-file generated/dut/MedCOT.jsonl --samples 3 --seed 42 --output results/ds-001/
```

Judge model is still **`--model`** on `batch` (default Haiku); configure `ANTHROPIC_API_KEY` unless you pass `--model minimax-m2.5` / `deepseek-chat`.

### 3. Single item (one question + one response)

```bash
python eval.py single \
  --task MedCOT \
  --dut YOUR_MODEL_NAME \
  --question "..." \
  --response "..." \
  [--gold-answer "..."]
```

### 4. Interactive wizard (human)

```bash
python eval.py
```

With **no arguments**, this starts prompts on stdin — **unsuitable for unattended agents**.

### Options you will need from docs

| Topic | Where to read |
|-------|----------------|
| Capability keys (`reasoning`, `long_context`, …) and task lists | `python eval.py tasks` and [`SKILL.md`](SKILL.md) |
| DUT file layouts (`--responses-dir`, `--responses-file`) | [`references/README.md`](references/README.md) |
| **`--calibrate-n`** / **`--calibrate-mode`** (few-shot anchors, BM25, etc.) | `--help` on `eval.py batch` and [`judge/README.md`](judge/README.md) |

---

## Judge alignment (validate)

Optional sanity check that the judge ranks gold above weak tiers (uses the API — not “offline”):

```bash
python -m scripts.validate --task MedCOT --samples 3
# or
python -m scripts.validate --all-tasks --samples 5
```

Same entry point: `python scripts/validate.py …`.

---

## Project layout

```
pyproject.toml          # project metadata + runtime deps
uv.lock                 # locked versions (commit this)
requirements.txt        # pip export of runtime deps (commit this)
tests/                  # pytest smoke (CLI + gold files + BM25 index)
eval.py                 # CLI: wizard | batch | single | tasks
scripts/
  validate.py           # alignment CLI (prefer: python -m scripts.validate)
judge/                  # refs | llm_client | scoring | runner — see judge/README.md
references/
  PROTOCOL.md           # what is “data/spec” vs executable code
  capabilities.json     # capability groups (wizard / --capability); edit for task lists
  rubrics.yaml          # task rubrics (source); edit criteria here
  rubrics.md            # generated from rubrics.yaml; do not edit by hand
  medbench-agent-95/    # gold JSONL per task
SKILL.md                # skill /medbench-eval behavior
```

---

## More documentation

| File | Contents |
|------|----------|
| [`references/README.md`](references/README.md) | Gold file format, rubric regeneration, skill-oriented notes |
| [`references/PROTOCOL.md`](references/PROTOCOL.md) | Protocol vs Python in this repo |
| [`judge/README.md`](judge/README.md) | Module map and implementation notes |

---

## Maintainers

- Commit **`pyproject.toml`**, **`uv.lock`**, and **`requirements.txt`** together when dependencies change.
- After editing the **`[project]` `dependencies`** list in `pyproject.toml`: run `uv lock`, then optionally `uv export …` for pip users.
- Regenerate **`references/rubrics.md`** after changing **`references/rubrics.yaml`**: `python -m judge.refs`.
- Edit **`references/capabilities.json`** to change capability → task grouping (no Python edit required for that).

---

## License

MIT
