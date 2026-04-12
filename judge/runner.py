"""
Batch evaluation pipeline: DUT file I/O, gold-corpus RAG, benchmark loop, reports.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from .llm_client import DEFAULT_MODEL
from .refs import get_tasks_for_capability, list_tasks
from .scoring import judge_against_gold, judge_response, JudgementResult

# ═══════════════════════════════════════════════════════════════════════════════
# DUT file I/O (JSONL / TXT / dirs)
# ═══════════════════════════════════════════════════════════════════════════════

ParsedResponses = dict[str, dict[int, str]]


def parse_responses(source: Union[str, Path]) -> ParsedResponses:
    """Auto-detect format and parse DUT responses from path."""
    path = Path(source)

    if path.is_dir():
        return _parse_benchmark_dir(path)

    text = path.read_text(encoding="utf-8")

    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    if first_line.startswith("{"):
        return _parse_jsonl(text, default_task="unknown")

    return _parse_delimited_txt(text)


def parse_responses_text(text: str, default_task: str = "unknown") -> ParsedResponses:
    """Parse DUT responses from raw pasted text (chat / clipboard).

    Tries JSONL when the first non-empty line looks like JSON; then delimited
    headers (``=== Task id ===``); then sequential chunks (order = sample
    order). For JSONL lines with ``id`` + ``response`` but no ``task``, uses
    ``default_task`` (match :func:`parse_responses` file JSONL when task is set
    per line).
    """
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")
    if first.startswith("{"):
        got = _parse_jsonl(text, default_task=default_task)
        if got:
            return got
    result = _parse_delimited_txt(text)
    if result:
        return result
    return _parse_sequential_txt(text, default_task)


def _parse_benchmark_dir(directory: Path) -> ParsedResponses:
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


def _parse_jsonl(text: str, default_task: str = "unknown") -> ParsedResponses:
    result: ParsedResponses = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "task" in rec and ("response" in rec or "answer" in rec):
            task = str(rec["task"])
            sample_id = int(rec.get("id", rec.get("question_id", 0)))
            response = str(rec.get("response", rec.get("answer", "")))
        elif "id" in rec and ("response" in rec or "answer" in rec):
            task = str(rec.get("task", default_task))
            sample_id = int(rec["id"])
            response = str(rec.get("response", rec.get("answer", "")))
        elif "other" in rec and isinstance(rec["other"], dict):
            other = rec["other"]
            source = str(other.get("source", ""))
            task = re.sub(r"_V\d+$", "", source) if source else "unknown"
            sample_id = int(other.get("id", 0))
            response = str(rec.get("answer", ""))
        else:
            continue

        if task and response:
            result.setdefault(task, {})[sample_id] = response

    return result


_HEADER_RE = re.compile(
    r"^(?:===|---|#+)\s*"
    r"([A-Za-z][A-Za-z0-9]+)"
    r"[\s|_\-]+(\d+)"
    r"[\s=\-]*$",
    re.MULTILINE,
)


def _parse_delimited_txt(text: str) -> ParsedResponses:
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


def _parse_sequential_txt(text: str, default_task: str) -> ParsedResponses:
    separator_re = re.compile(r"^={3,}$", re.MULTILINE)
    chunks = separator_re.split(text)
    if len(chunks) == 1:
        chunks = re.split(r"\n{2,}", text.strip())

    result: ParsedResponses = {}
    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if chunk:
            result.setdefault(default_task, {})[idx] = chunk

    return result


def responses_to_jsonl(responses: ParsedResponses) -> str:
    lines = []
    for task, samples in sorted(responses.items()):
        for sample_id, response in sorted(samples.items()):
            lines.append(json.dumps(
                {"task": task, "id": sample_id, "response": response},
                ensure_ascii=False,
            ))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# RAG over shipped gold JSONL (BM25 / embedding)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GoldSample:
    sample_id: int
    question: str
    answer: str


def _tokenize(text: str) -> list[str]:
    tokens = []
    buf = []
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f':
            if buf:
                tokens.append("".join(buf).lower())
                buf = []
            tokens.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf).lower())
                buf = []
    if buf:
        tokens.append("".join(buf).lower())
    return tokens


class _BM25Index:
    K1 = 1.5
    B = 0.75

    def __init__(self, docs: list[str]) -> None:
        self.docs = docs
        self.n = len(docs)
        self.tokenized = [_tokenize(d) for d in docs]
        self.avg_dl = sum(len(t) for t in self.tokenized) / max(self.n, 1)
        self.df: dict[str, int] = Counter()
        for tokens in self.tokenized:
            for term in set(tokens):
                self.df[term] += 1

    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = _tokenize(query)
        doc_tokens = self.tokenized[doc_idx]
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in self.df:
                continue
            idf = math.log((self.n - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
            tf_norm = (tf[term] * (self.K1 + 1)) / (
                tf[term] + self.K1 * (1 - self.B + self.B * dl / self.avg_dl)
            )
            score += idf * tf_norm
        return score

    def top_n(self, query: str, n: int, exclude: set[int]) -> list[int]:
        scores = [
            (i, self.score(query, i))
            for i in range(self.n)
            if i not in exclude
        ]
        scores.sort(key=lambda x: -x[1])
        return [i for i, _ in scores[:n]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def _embed(texts: list[str], api_key: str, base_url: str) -> list[list[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    payload = json.dumps({
        "model": "BAAI/bge-m3",
        "input": texts,
        "encoding_format": "float",
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Embedding API error {e.code}: {e.read().decode()}") from e
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


class GoldAnchorIndex:
    """BM25 or embedding retrieval over one task's gold JSONL."""

    def __init__(
        self,
        benchmark_dir: str | Path,
        task: str,
        backend: str = "bm25",
        embed_field: str = "question",
    ) -> None:
        self.task = task
        self.backend = backend
        self.embed_field = embed_field

        self._samples: list[GoldSample] = self._load(Path(benchmark_dir), task)
        self._doc_texts: list[str] = [self._doc_text(s) for s in self._samples]

        if backend == "bm25":
            self._bm25 = _BM25Index(self._doc_texts)
            self._embeddings: list[list[float]] = []
        elif backend == "embedding":
            api_key = os.environ.get("MINIMAX_API_KEY", "")
            base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.siliconflow.cn/v1")
            if not api_key:
                raise EnvironmentError("MINIMAX_API_KEY not set (needed for embedding backend)")
            self._api_key = api_key
            self._base_url = base_url
            self._embeddings = _embed(self._doc_texts, api_key, base_url)
            self._bm25 = _BM25Index([])
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'bm25' or 'embedding'.")

    def _doc_text(self, s: GoldSample) -> str:
        if self.embed_field == "answer":
            return s.answer
        if self.embed_field == "both":
            return s.question + " " + s.answer
        return s.question

    @staticmethod
    def _load(benchmark_dir: Path, task: str) -> list[GoldSample]:
        path = benchmark_dir / f"{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                other = rec.get("other", {})
                sid = other.get("id", len(samples)) if isinstance(other, dict) else len(samples)
                q = str(rec.get("question", ""))
                a = str(rec.get("answer", ""))
                if q and a:
                    samples.append(GoldSample(sample_id=int(sid), question=q, answer=a))
        return samples

    def retrieve(
        self,
        question: str,
        n: int = 2,
        exclude_ids: Optional[set[int]] = None,
    ) -> list[dict]:
        exclude_ids = exclude_ids or set()
        excluded_indices = {
            i for i, s in enumerate(self._samples) if s.sample_id in exclude_ids
        }

        if self.backend == "bm25":
            top_idx = self._bm25.top_n(question, n, excluded_indices)
        else:
            query_emb = _embed([question], self._api_key, self._base_url)[0]
            scored = [
                (i, _cosine(query_emb, self._embeddings[i]))
                for i in range(len(self._samples))
                if i not in excluded_indices
            ]
            scored.sort(key=lambda x: -x[1])
            top_idx = [i for i, _ in scored[:n]]

        return [
            {"question": self._samples[i].question, "answer": self._samples[i].answer}
            for i in top_idx
        ]

    def __len__(self) -> int:
        return len(self._samples)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark loop
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARK_TASKS = list_tasks()


def _load_anchor_examples(
    benchmark_dir: Path,
    task: str,
    n: int = 2,
    seed: int = 99,
    exclude_ids: Optional[set] = None,
) -> list[dict]:
    samples = []
    jsonl_path = benchmark_dir / f"{task}.jsonl"
    if not jsonl_path.exists():
        return []

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
            sid = other.get("id") if isinstance(other, dict) else None
            if exclude_ids and sid in exclude_ids:
                continue
            q = str(rec.get("question", ""))
            a = str(rec.get("answer", ""))
            if q and a:
                samples.append({"question": q, "answer": a})

    rng = random.Random(seed)
    return rng.sample(samples, min(n, len(samples)))


@dataclass
class SampleResult:
    task: str
    sample_id: int
    source: str
    question: str
    gold_answer: str
    evaluated_response: str
    judgement: JudgementResult
    is_gold_eval: bool = False


@dataclass
class BenchmarkResult:
    task_results: dict[str, list[SampleResult]] = field(default_factory=dict)
    model: str = DEFAULT_MODEL
    benchmark_dir: str = ""
    errors: list[str] = field(default_factory=list)


def _load_task_samples(benchmark_dir: Path, task: str) -> list[dict]:
    jsonl_path = benchmark_dir / f"{task}.jsonl"
    if not jsonl_path.exists():
        return []

    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


def load_task_jsonl_samples(benchmark_dir: Union[str, Path], task: str) -> list[dict]:
    """Load all records from ``benchmark_dir / f'{task}.jsonl'``."""
    return _load_task_samples(Path(benchmark_dir), task)


def run_benchmark(
    benchmark_dir: str,
    tasks: Optional[list[str]] = None,
    capability: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    dut: str = "unknown",
    samples_per_task: int = 5,
    evaluate_gold: bool = False,
    responses_dir: Optional[str] = None,
    responses_file: Optional[str] = None,
    calibrate_n: int = 0,
    calibrate_mode: str = "random",
    seed: int = 42,
    delay_seconds: float = 0.5,
    verbose: bool = True,
) -> BenchmarkResult:
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        raise ValueError(f"Benchmark directory not found: {benchmark_dir}")

    if capability:
        tasks_to_run = list(get_tasks_for_capability(capability))
    else:
        tasks_to_run = tasks or BENCHMARK_TASKS
    tasks_to_run = [t for t in tasks_to_run if (benchmark_path / f"{t}.jsonl").exists()]

    parsed_responses: Optional[ParsedResponses] = None
    if responses_file:
        parsed_responses = parse_responses(responses_file)
        if verbose:
            total = sum(len(v) for v in parsed_responses.values())
            print(f"Loaded {total} DUT responses from: {responses_file}")

    result = BenchmarkResult(model=model, benchmark_dir=str(benchmark_path))
    rng = random.Random(seed)

    for task in tasks_to_run:
        if verbose:
            print(f"\n[{task}] Loading samples...")

        samples = _load_task_samples(benchmark_path, task)
        if not samples:
            result.errors.append(f"{task}: no samples found")
            continue

        selected = rng.sample(samples, min(samples_per_task, len(samples)))
        selected_ids = {
            s.get("other", {}).get("id") for s in selected
            if isinstance(s.get("other"), dict)
        }
        task_results = []

        gold_anchor_index: Optional[GoldAnchorIndex] = None
        anchor_examples_static: Optional[list[dict]] = None
        if calibrate_n > 0:
            if calibrate_mode == "random":
                anchor_examples_static = _load_anchor_examples(
                    benchmark_path, task, n=calibrate_n, seed=seed, exclude_ids=selected_ids
                )
                if verbose and anchor_examples_static:
                    print(f"  [{task}] calibrating with {len(anchor_examples_static)} gold anchors (random)")
            else:
                try:
                    gold_anchor_index = GoldAnchorIndex(
                        benchmark_path, task, backend=calibrate_mode
                    )
                    if verbose:
                        print(
                            f"  [{task}] gold anchor index ready ({calibrate_mode}, "
                            f"{len(gold_anchor_index)} samples)"
                        )
                except Exception as e:
                    if verbose:
                        print(
                            f"  [{task}] gold anchor index init failed ({e}), "
                            f"falling back to random anchors"
                        )
                    anchor_examples_static = _load_anchor_examples(
                        benchmark_path, task, n=calibrate_n, seed=seed, exclude_ids=selected_ids
                    )

        for i, sample in enumerate(selected):
            question = str(sample.get("question", ""))
            gold_answer = str(sample.get("answer", ""))
            other = sample.get("other", {})
            sample_id = other.get("id", i) if isinstance(other, dict) else i
            source = other.get("source", task) if isinstance(other, dict) else task

            if evaluate_gold:
                evaluated_response = gold_answer
                is_gold = True
            elif parsed_responses:
                task_resp = parsed_responses.get(task, {})
                evaluated_response = task_resp.get(int(sample_id), "")
                if not evaluated_response:
                    result.errors.append(f"{task}/{sample_id}: not found in responses file")
                    continue
                is_gold = False
            elif responses_dir:
                resp_path = Path(responses_dir) / task / f"{sample_id}.txt"
                if resp_path.exists():
                    evaluated_response = resp_path.read_text(encoding="utf-8")
                else:
                    result.errors.append(f"{task}/{sample_id}: response file not found at {resp_path}")
                    continue
                is_gold = False
            else:
                evaluated_response = gold_answer
                is_gold = True

            anchor_examples: Optional[list[dict]] = None
            if calibrate_n > 0:
                if gold_anchor_index is not None:
                    anchor_examples = gold_anchor_index.retrieve(
                        question=question,
                        n=calibrate_n,
                        exclude_ids={int(sample_id)},
                    )
                else:
                    anchor_examples = anchor_examples_static

            if verbose:
                mode = "gold" if is_gold else "model"
                print(f"  [{i+1}/{len(selected)}] {task} sample {sample_id} ({mode})...")

            if is_gold:
                judgement = judge_against_gold(
                    task=task,
                    question=question,
                    gold_answer=gold_answer,
                    model=model,
                    dut=dut,
                    anchor_examples=anchor_examples,
                )
            else:
                judgement = judge_response(
                    task=task,
                    question=question,
                    response=evaluated_response,
                    gold_answer=gold_answer,
                    model=model,
                    dut=dut,
                    anchor_examples=anchor_examples,
                )

            if judgement.error and verbose:
                print(f"    ERROR: {judgement.error}")

            task_results.append(SampleResult(
                task=task,
                sample_id=sample_id,
                source=source,
                question=question,
                gold_answer=gold_answer,
                evaluated_response=evaluated_response,
                judgement=judgement,
                is_gold_eval=is_gold,
            ))

            if delay_seconds > 0 and i < len(selected) - 1:
                time.sleep(delay_seconds)

        result.task_results[task] = task_results
        if verbose and task_results:
            avg = sum(r.judgement.normalized_score for r in task_results) / len(task_results)
            print(f"  [{task}] avg score: {avg:.1f}/100")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Reports (summary JSON, details JSONL, markdown)
# ═══════════════════════════════════════════════════════════════════════════════


def _task_stats(results: list[SampleResult]) -> dict[str, Any]:
    if not results:
        return {"n": 0, "avg_score": 0.0, "min_score": 0.0, "max_score": 0.0, "errors": 0,
                "total_minor_errors": 0, "total_major_errors": 0}

    scores = [r.judgement.normalized_score for r in results if not r.judgement.error]
    errors = sum(1 for r in results if r.judgement.error)

    criterion_avgs: dict[str, list[float]] = {}
    for r in results:
        for cs in r.judgement.criterion_scores:
            criterion_avgs.setdefault(cs.name, []).append(cs.score)

    criterion_means = {
        k: round(sum(v) / len(v), 3) for k, v in criterion_avgs.items()
    }

    total_minor = sum(len(r.judgement.minor_errors) for r in results)
    total_major = sum(len(r.judgement.major_errors) for r in results)

    return {
        "n": len(results),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min_score": round(min(scores), 2) if scores else 0.0,
        "max_score": round(max(scores), 2) if scores else 0.0,
        "errors": errors,
        "total_minor_errors": total_minor,
        "total_major_errors": total_major,
        "criterion_means": criterion_means,
    }


def build_summary(result: BenchmarkResult) -> dict[str, Any]:
    task_stats: dict[str, Any] = {}
    all_scores = []

    for task, results in result.task_results.items():
        stats = _task_stats(results)
        task_stats[task] = stats
        if stats["n"] > 0:
            all_scores.extend(
                r.judgement.normalized_score
                for r in results
                if not r.judgement.error
            )

    overall_avg = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0

    dut = "unknown"
    for task_results in result.task_results.values():
        for r in task_results:
            if r.judgement.dut:
                dut = r.judgement.dut
                break
        if dut != "unknown":
            break

    return {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "judge_model": result.model,
            "dut": dut,
            "benchmark_dir": result.benchmark_dir,
            "tasks_evaluated": list(result.task_results.keys()),
            "total_samples": sum(s["n"] for s in task_stats.values()),
            "total_errors": sum(s["errors"] for s in task_stats.values()),
            "total_minor_errors": sum(s.get("total_minor_errors", 0) for s in task_stats.values()),
            "total_major_errors": sum(s.get("total_major_errors", 0) for s in task_stats.values()),
        },
        "overall": {
            "avg_score": overall_avg,
            "score_range": "0-100 (normalized from 1-5 Likert)",
        },
        "tasks": task_stats,
        "errors": result.errors,
    }


def save_results(result: BenchmarkResult, output_dir: str) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = build_summary(result)

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    details_path = out / "details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for task, task_results in result.task_results.items():
            for r in task_results:
                record = {
                    "task": r.task,
                    "sample_id": r.sample_id,
                    "source": r.source,
                    "dut": r.judgement.dut,
                    "is_gold_eval": r.is_gold_eval,
                    "normalized_score": r.judgement.normalized_score,
                    "total_score": r.judgement.total_score,
                    "criteria": [
                        {"name": cs.name, "score": cs.score, "justification": cs.justification}
                        for cs in r.judgement.criterion_scores
                    ],
                    "overall_feedback": r.judgement.overall_feedback,
                    "minor_errors": r.judgement.minor_errors,
                    "major_errors": r.judgement.major_errors,
                    "error": r.judgement.error,
                    "tokens": {
                        "input": r.judgement.input_tokens,
                        "output": r.judgement.output_tokens,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    md_path = out / "report.md"
    md_path.write_text(_build_markdown(summary, result), encoding="utf-8")

    return {
        "summary": str(summary_path),
        "details": str(details_path),
        "report": str(md_path),
    }


def _build_markdown(summary: dict[str, Any], result: BenchmarkResult) -> str:
    meta = summary["meta"]
    overall = summary["overall"]
    tasks = summary["tasks"]

    lines = [
        "# MedBench-Eval Report",
        "",
        f"**Timestamp:** {meta['timestamp']}  ",
        f"**DUT (Model Under Test):** `{meta['dut']}`  ",
        f"**Judge Model:** `{meta['judge_model']}`  ",
        f"**Benchmark:** `{meta['benchmark_dir']}`  ",
        f"**Total Samples:** {meta['total_samples']}  ",
        f"**Total Errors:** {meta['total_errors']}  ",
        f"**Minor Errors (no score impact):** {meta.get('total_minor_errors', 0)}  ",
        f"**Major Errors (score impact):** {meta.get('total_major_errors', 0)}",
        "",
        "## Overall",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Average Score | **{overall['avg_score']}/100** |",
        f"| Score Range | {overall['score_range']} |",
        "",
        "## Per-Task Results",
        "",
        "| Task | N | Avg Score | Min | Max | Minor Err | Major Err | Judge Err |",
        "|------|---|-----------|-----|-----|-----------|-----------|-----------|",
    ]

    for task, stats in sorted(tasks.items()):
        lines.append(
            f"| {task} | {stats['n']} | {stats['avg_score']:.1f} | "
            f"{stats['min_score']:.1f} | {stats['max_score']:.1f} | "
            f"{stats.get('total_minor_errors', 0)} | "
            f"{stats.get('total_major_errors', 0)} | "
            f"{stats['errors']} |"
        )

    lines += ["", "## Per-Criterion Breakdown", ""]

    for task, stats in sorted(tasks.items()):
        if not stats.get("criterion_means"):
            continue
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| Criterion | Avg Score (1-5) |")
        lines.append("|-----------|-----------------|")
        for crit, mean in stats["criterion_means"].items():
            lines.append(f"| {crit} | {mean:.2f} |")
        lines.append("")

    if result.errors:
        lines += ["## Errors", ""]
        for err in result.errors:
            lines.append(f"- {err}")
        lines.append("")

    lines += [
        "---",
        "*Generated by [medbench-eval](https://github.com/chenhao/medqa-auto-eval-skills)*",
    ]

    return "\n".join(lines)


def print_summary(result: BenchmarkResult) -> None:
    summary = build_summary(result)
    overall = summary["overall"]
    tasks = summary["tasks"]

    print("\n" + "=" * 60)
    print("MedBench-Eval Results")
    print("=" * 60)
    print(f"Judge model : {result.model}")
    print(f"Overall avg : {overall['avg_score']:.1f}/100")
    print()
    print(f"{'Task':<18} {'N':>4} {'Avg':>8} {'Min':>8} {'Max':>8} {'Err':>5}")
    print("-" * 60)
    for task, stats in sorted(tasks.items()):
        print(
            f"{task:<18} {stats['n']:>4} "
            f"{stats['avg_score']:>8.1f} {stats['min_score']:>8.1f} "
            f"{stats['max_score']:>8.1f} {stats['errors']:>5}"
        )
    print("=" * 60)
