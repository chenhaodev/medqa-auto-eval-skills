"""
RAG-based gold anchor retrieval for LLM-as-judge calibration.

Instead of randomly sampling gold anchor examples, this module retrieves the
gold examples that are MOST SEMANTICALLY SIMILAR to the question being evaluated.

For MedCollab and other task types where the calibration example's clinical
domain matters, semantic retrieval dramatically improves judge alignment:

  Random: "Evaluate drug interaction question" → show lung cancer orchestration example
  RAG:    "Evaluate drug interaction question" → show drug ADR orchestration example

Two similarity backends:

  "bm25"      — character-level BM25 (default, zero dependencies, works well for Chinese)
  "embedding" — SiliconFlow BAAI/bge-m3 embeddings (better quality, needs SILICONFLOW key)
               Env vars: MINIMAX_API_KEY (reused) + MINIMAX_BASE_URL

Usage:
    from judge.rag import GoldRAG

    rag = GoldRAG(benchmark_dir="medbench-agent-95", task="MedCollab")
    examples = rag.retrieve(question=current_question, n=2, exclude_ids={97, 31})
    # returns [{"question": str, "answer": str}, ...]
"""

import json
import math
import os
import urllib.request
import urllib.error
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class GoldSample:
    sample_id: int
    question: str
    answer: str


# ── BM25 implementation (zero dependencies) ───────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Character-level tokenization suitable for Chinese + Latin mixed text.
    Each CJK character and each ASCII word is a token.
    """
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
    """Minimal BM25 index over a list of documents."""

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


# ── Embedding backend (SiliconFlow BAAI/bge-m3) ───────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def _embed(texts: list[str], api_key: str, base_url: str) -> list[list[float]]:
    """Call SiliconFlow embedding API (OpenAI-compatible /v1/embeddings)."""
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
    # Sort by index to guarantee order
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


# ── Main class ────────────────────────────────────────────────────────────────

class GoldRAG:
    """
    Retrieves the most semantically similar gold examples for a given question.

    Args:
        benchmark_dir: Path to medbench-agent-95/ directory
        task: Task name (e.g. "MedCollab")
        backend: "bm25" (default) or "embedding"
        embed_field: Which field to embed/index — "question" (default), "answer", or "both"
    """

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
            self._bm25 = _BM25Index([])  # unused
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'bm25' or 'embedding'.")

    def _doc_text(self, s: GoldSample) -> str:
        if self.embed_field == "answer":
            return s.answer
        if self.embed_field == "both":
            return s.question + " " + s.answer
        return s.question  # default

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
        """
        Retrieve the N most similar gold examples for the given question.

        Args:
            question: The clinical question being evaluated
            n: Number of examples to return
            exclude_ids: Sample IDs to exclude (typically the evaluated sample's ID)

        Returns:
            List of {"question": str, "answer": str} dicts, best match first
        """
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
