# `references/` — skill-side materials

Matches the usual Agent Skill layout: `SKILL.md` is the entry; this folder holds **human-readable notes**, **protocol data** (`capabilities.json`), and **gold-standard data** consumed by `eval.py` / `judge/*` at runtime (do not load entire trees into model context). See [`PROTOCOL.md`](PROTOCOL.md) for what is spec vs executable code.

## `rubrics.yaml` / `rubrics.md`

- **Source of truth for criteria:** [`rubrics.yaml`](rubrics.yaml) — edit task descriptions and criterion anchors here.
- **`rubrics.md`** is **generated** (`python -m judge.refs`) for human/model reading — do not edit by hand.
- When judging, use per-task **criterion names** and Score **1/5 anchors**; same semantics as the CLI judge.

## `medbench-agent-95/` (gold benchmark)

- **MedBench-Agent-95:** 13 tasks, one `{Task}.jsonl` per task (e.g. `MedCOT.jsonl`), one JSON object per line.
- **Fields** (OpenCompass / MedBench export): `question`, `answer` (gold response), `other` (includes `id`, `source`, etc.).
- **Used by:**
  - `judge/runner.py` — batch sampling/scoring, DUT file I/O, and `GoldAnchorIndex` (BM25/embedding) for `--calibrate-n`;
  - `judge/scoring.py` — `gold_answer` calibrates “what good looks like”; the DUT is not required to match verbatim.
- **Default CLI path:** `references/medbench-agent-95` (`eval.py batch` default `--benchmark`; see `judge/refs.py`).

**Interactive skill:** when showing items to the user, open **one JSONL row by sample id** — do not paste the whole directory into the chat.
