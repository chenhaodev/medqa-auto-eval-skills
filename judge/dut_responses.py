"""
Parse DUT (Device Under Test) responses from multiple input formats.

Supported formats:

  FORMAT 1 — Benchmark JSONL (same schema as references/medbench-agent-95/*.jsonl)
    Each line: {"question": "...", "answer": "<DUT response>", "other": {"id": 97, ...}}
    Use when: you have a full results directory matching the benchmark structure.

  FORMAT 2 — Compact JSONL (lightweight, machine-generated)
    Each line: {"task": "MedCOT", "id": 97, "response": "<DUT response>"}
    Use when: your DUT pipeline produces structured output per sample.

  FORMAT 3 — Delimited TXT (human-readable, easy to produce manually)
    Section header: === MedCOT | 97 ===  (or === MedCOT_97 === or # MedCOT 97)
    Body: raw DUT response text (multi-line OK)
    Use when: you paste/concatenate DUT outputs from a chat session.

  FORMAT 4 — Sequential TXT (no IDs, relies on question order)
    Responses separated by blank lines or === separators.
    Must be paired with a task name and the benchmark JSONL to map IDs by order.
    Use when: DUT output is completely unstructured.

All parsers return: dict[task_name, dict[sample_id, response_text]]
"""

import json
import re
from pathlib import Path
from typing import Union


ParsedResponses = dict[str, dict[int, str]]   # {task: {id: response}}


# ── Public API ────────────────────────────────────────────────────────────────

def parse_responses(source: Union[str, Path]) -> ParsedResponses:
    """
    Auto-detect format and parse DUT responses.

    Args:
        source: Path to a JSONL file, a TXT file, or a directory of JSONL files.

    Returns:
        {task_name: {sample_id: response_text}}
    """
    path = Path(source)

    if path.is_dir():
        return _parse_benchmark_dir(path)

    text = path.read_text(encoding="utf-8")

    # Detect JSONL: first non-blank line starts with '{'
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    if first_line.startswith("{"):
        return _parse_jsonl(text)

    # Fall back to delimited TXT
    return _parse_delimited_txt(text)


def parse_responses_text(text: str, default_task: str = "unknown") -> ParsedResponses:
    """
    Parse DUT responses from raw text (e.g. pasted into a chat window).
    Tries delimited TXT first, then sequential TXT as fallback.
    """
    result = _parse_delimited_txt(text)
    if result:
        return result
    # Sequential fallback — all responses go under default_task with sequential IDs
    return _parse_sequential_txt(text, default_task)


# ── Format 1: benchmark directory ────────────────────────────────────────────

def _parse_benchmark_dir(directory: Path) -> ParsedResponses:
    """Parse a directory of JSONL files (same schema as references/medbench-agent-95/)."""
    result: ParsedResponses = {}
    for jsonl_path in sorted(directory.glob("*.jsonl")):
        task = jsonl_path.stem
        result[task] = {}
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                other = rec.get("other", {})
                sample_id = other.get("id") if isinstance(other, dict) else None
                response = str(rec.get("answer", ""))
                if sample_id is not None and response:
                    result[task][int(sample_id)] = response
    return {k: v for k, v in result.items() if v}


# ── Format 2: compact JSONL ───────────────────────────────────────────────────

def _parse_jsonl(text: str) -> ParsedResponses:
    """
    Parse JSONL text.  Supports two schemas:
      - Compact:   {"task": "MedCOT", "id": 97, "response": "..."}
      - Benchmark: {"question": "...", "answer": "...", "other": {"id": 97, "source": "MedCOT_V4"}}
    """
    result: ParsedResponses = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Compact format
        if "task" in rec and ("response" in rec or "answer" in rec):
            task = str(rec["task"])
            sample_id = int(rec.get("id", rec.get("question_id", 0)))
            response = str(rec.get("response", rec.get("answer", "")))
        # Benchmark format — derive task from "source" field
        elif "other" in rec and isinstance(rec["other"], dict):
            other = rec["other"]
            source = str(other.get("source", ""))
            # source examples: "MedCOT_V4", "MedCallAPI_V2" → strip _V\d+
            task = re.sub(r"_V\d+$", "", source) if source else "unknown"
            sample_id = int(other.get("id", 0))
            response = str(rec.get("answer", ""))
        else:
            continue

        if task and response:
            result.setdefault(task, {})[sample_id] = response

    return result


# ── Format 3: delimited TXT ───────────────────────────────────────────────────

# Patterns for section headers like:
#   === MedCOT | 97 ===
#   === MedCOT_97 ===
#   # MedCOT 97
#   --- MedCOT | 97 ---
_HEADER_RE = re.compile(
    r"^(?:===|---|#+)\s*"          # opener
    r"([A-Za-z][A-Za-z0-9]+)"     # task name
    r"[\s|_\-]+(\d+)"             # separator + id
    r"[\s=\-]*$",                  # closer
    re.MULTILINE,
)


def _parse_delimited_txt(text: str) -> ParsedResponses:
    """
    Parse delimited TXT.  Section header: === <Task> | <id> ===
    Everything between two headers is the DUT response for that sample.
    """
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return {}

    result: ParsedResponses = {}
    for i, match in enumerate(matches):
        task = match.group(1)
        sample_id = int(match.group(2))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        response = text[start:end].strip()
        if response:
            result.setdefault(task, {})[sample_id] = response

    return result


# ── Format 4: sequential TXT ─────────────────────────────────────────────────

def _parse_sequential_txt(text: str, default_task: str) -> ParsedResponses:
    """
    Parse sequentially ordered responses separated by blank lines or === lines.
    Assigns sequential integer IDs starting from 0.
    """
    separator_re = re.compile(r"^={3,}$", re.MULTILINE)
    # Split on either === separators or 2+ consecutive blank lines
    chunks = separator_re.split(text)
    if len(chunks) == 1:
        chunks = re.split(r"\n{2,}", text.strip())

    result: ParsedResponses = {}
    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if chunk:
            result.setdefault(default_task, {})[idx] = chunk

    return result


# ── Utility ───────────────────────────────────────────────────────────────────

def responses_to_jsonl(responses: ParsedResponses) -> str:
    """Serialise parsed responses back to compact JSONL for inspection."""
    lines = []
    for task, samples in sorted(responses.items()):
        for sample_id, response in sorted(samples.items()):
            lines.append(json.dumps(
                {"task": task, "id": sample_id, "response": response},
                ensure_ascii=False,
            ))
    return "\n".join(lines)
