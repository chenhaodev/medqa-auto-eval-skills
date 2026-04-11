# `references/` — skill-side materials

Matches the usual Agent Skill layout: `SKILL.md` is the entry; this folder holds **human-readable notes** and **gold-standard data** consumed by `eval.py` / `judge/*` at runtime (do not load entire trees into model context).

## `rubrics.md`

- **Generated** from `judge/rubrics.py` via `python scripts/export_rubrics_md.py` — do not edit by hand.
- When judging, use per-task **criterion names** and Score **1/5 anchors**; same semantics as the CLI judge.

## `medbench-agent-95/` (gold benchmark)

- **MedBench-Agent-95:** 13 tasks, one `{Task}.jsonl` per task (e.g. `MedCOT.jsonl`), one JSON object per line.
- **Fields** (OpenCompass / MedBench export): `question`, `answer` (gold response), `other` (includes `id`, `source`, etc.).
- **Used by:**
  - `judge/batch_runner.py` — sampling and batch scoring;
  - `judge/gold_retrieval.py` — BM25 / embedding over gold text for `--calibrate-n` few-shot anchors;
  - `judge/scoring.py` — `gold_answer` calibrates “what good looks like”; the DUT is not required to match verbatim.
- **Default CLI path:** `references/medbench-agent-95` (`eval.py batch` default `--benchmark`; see `judge/paths.py`).

**Interactive skill:** when showing items to the user, open **one JSONL row by sample id** — do not paste the whole directory into the chat.
