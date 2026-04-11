# medbench-eval

**LLM-as-judge** on the **MedBench-Agent-95** benchmark (390 items, 13 task types). Per-criterion Likert 1–5, normalized to **0–100**.

**Entry points:** `/medbench-eval` via `SKILL.md` (interactive); `eval.py` for batch/single automation. Rubric anchors: `references/rubrics.md` (generated — run `python scripts/export_rubrics_md.py` after editing `judge/rubrics.py`). **Gold JSONL:** `references/medbench-agent-95/` — see [`references/README.md`](references/README.md).

---

## Install

Python **3.11+**. Set `ANTHROPIC_API_KEY` for the default judge (`claude-haiku-4-5`). Optional: `MINIMAX_*` / `DEEPSEEK_*` (see [`judge/llm_client.py`](judge/llm_client.py)). A root `.env` is auto-loaded.

**Using [uv](https://docs.astral.sh/uv/) (recommended):** from the repo root, run `uv sync`. That creates `.venv/` and installs dependencies from **`uv.lock`**.

**Using pip:** `pip install -r requirements.txt` — that file is **generated** from the lockfile:

```bash
uv export --format requirements-txt -o requirements.txt
```

After changing **`pyproject.toml`** `[project.dependencies]`, run **`uv lock`** then optionally re-export `requirements.txt` for pip users.

**Version control:** commit **`pyproject.toml`**, **`uv.lock`**, and **`requirements.txt`** together so installs are reproducible. **`uv.lock`** is the source of truth for resolved versions; **`requirements.txt`** is an export for `pip` users.

---

## For coding agents (Cursor, Claude Code, …)

Use this when you **automate** setup from a clean environment (no interactive `eval.py` wizard). Replace `<repo-url>` with the real Git remote.

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd <checkout-directory>   # repository root: must contain pyproject.toml
   ```
   (If you only have a tarball, unpack it and `cd` into that root.)

2. **Install uv** (if `uv` is not already on `PATH`):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Follow the installer’s note to reload your shell or add `~/.local/bin` to `PATH`.

3. **Install dependencies** (creates **`.venv/`** from **`uv.lock`**; includes **pytest** for checks):
   ```bash
   uv sync --group dev
   ```

4. **Configure API keys** — create **`.env`** in the repo root (do **not** commit it):
   ```bash
   printf '%s\n' 'ANTHROPIC_API_KEY=sk-ant-api03-...' > .env
   ```
   Optional env vars for non-default judge backends: see [`judge/llm_client.py`](judge/llm_client.py).

5. **Run offline smoke tests** (no network to model APIs):
   ```bash
   uv run --group dev pytest tests/ -q
   python -m scripts.validate --help
   python eval.py tasks
   ```

6. **Operate** — read [`SKILL.md`](SKILL.md) for `/medbench-eval` prompt conventions; use `python eval.py batch …` or `python eval.py single …` for real judging (**requires** a valid `ANTHROPIC_API_KEY` or another configured backend).

`pip install -r requirements.txt` installs **runtime** dependencies only; use **`uv sync --group dev`** if you need **pytest** and the lockfile-resolved dev toolchain.

---

## Quick start

```bash
python eval.py tasks
python eval.py

python eval.py batch \
  --capability reasoning --dut YOUR_MODEL \
  --responses-dir outputs/YOUR_MODEL/ --samples 5 --output results/

python eval.py single --task MedCOT --dut YOUR_MODEL \
  --question "..." --response "..." [--gold-answer "..."]
```

`--benchmark` defaults to the shipped **`references/medbench-agent-95`** (absolute path resolved from repo root via `judge/paths.py`).

**Capability keys:** `reasoning`, `long_context`, `tool_use`, `orchestration`, `self_correction`, `role_adapt`, `safety`, `full`.

**DUT inputs:** `--responses-dir` as `{dir}/{Task}/{id}.txt`, or `--responses-file` as compact JSONL, full benchmark JSONL, or `=== Task | id ===` delimited TXT.

**Calibration:** `--calibrate-n 2` and optionally `--calibrate-mode bm25` (often helps MedCollab, MedDBOps, MedShield). Retrieval indexes gold text from the same benchmark directory.

**Batch output:** `summary.json`, `details.jsonl`, `report.md` under `--output`.

---

## Judge alignment (validate)

From the **repository root**:

```bash
python -m scripts.validate --all-tasks --samples 5
```

Equivalent: `python scripts/validate.py ...` (same module). Default `--benchmark` points at `references/medbench-agent-95`.

---

## Documentation (English)

| Doc | Contents |
|-----|----------|
| [`references/README.md`](references/README.md) | Rubrics generation, gold benchmark layout, skill usage |
| [`judge/README.md`](judge/README.md) | Python package: modules and dependency sketch |

---

## Layout

```
pyproject.toml, uv.lock       # dependencies (uv); requirements.txt exported for pip
tests/                        # pytest E2E smoke (CLI + offline gold index)
eval.py                       # CLI: wizard, batch, single, tasks
scripts/
  export_rubrics_md.py        # regenerate references/rubrics.md
  validate.py                 # judge–human alignment (run as module, see above)
judge/                        # see judge/README.md
references/
  README.md
  rubrics.md                  # generated
  medbench-agent-95/*.jsonl   # gold standard
SKILL.md
```

---

## License

MIT
