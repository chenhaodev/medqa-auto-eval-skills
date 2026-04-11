# Protocol vs code

This repository mixes **human-editable specifications** with **Python that must run** (API calls, batch loops, parsing). The goal is to keep **static policy** out of `.py` where possible so TUI / skill workflows do not depend on “reading the codebase.”

| Artifact | Role |
|----------|------|
| [`SKILL.md`](../SKILL.md) | In-chat judging protocol for `/medbench-eval` (no Python required). |
| [`capabilities.json`](capabilities.json) | Capability groups → task lists for the wizard and `eval.py batch --capability`. Edit this JSON to change grouping; `judge/capabilities.py` only loads it. |
| [`rubrics.yaml`](rubrics.yaml) | **Source** for task rubrics and criteria (Likert anchors). |
| [`rubrics.md`](rubrics.md) | Human-readable copy for models (generated: `python -m judge.refs`). |
| [`medbench-agent-95/`](medbench-agent-95/) | Gold questions/answers (JSONL). Data only. |

**Why Python remains:** calling judge LLMs (`judge/llm_client.py`), scoring prompts (`judge/scoring.py`), and the combined batch pipeline (`judge/runner.py`: benchmark walk, DUT file I/O, gold RAG, reports), plus optional alignment (`scripts/validate.py`), are **not** replaceable by markdown alone without losing automation.

**History:** rubric text used to live in standalone `judge/rubrics.py`; it now loads from **`references/rubrics.yaml`** via **`judge/refs.py`**. Export: `python -m judge.refs`.
