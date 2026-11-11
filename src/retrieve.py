"""Load the FAISS index and retrieve the top-k chunks for a question."""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from src.ingest import CHUNKS_FILE, INDEX_DIR, INDEX_FILE, Chunk, embed_texts

logger = logging.getLogger(__name__)

TOP_K = 4
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
MODES = ("hybrid", "vector", "keyword")

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was were what when "
    "where which who why will with do does did can".split()
)


class IndexNotFoundError(RuntimeError):
    """Index missing or empty."""


@dataclass
class RetrievedChunk:
    chunk_id: int
    source: str
    page: int
    text: str
    score: float


_HINT = "Run `uv run python -m src.ingest` to build it."


def tokenize(text: str) -> list[str]:
    """Lowercase, split into alphanumeric tokens (hyphenated terms kept whole), drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class _BM25:
    """Precomputed BM25 statistics over the chunk texts."""

    tfs: list[Counter]
    lengths: np.ndarray
    avgdl: float
    idf: dict[str, float]


def _build_bm25(chunks: list[Chunk]) -> _BM25:
    tfs = [Counter(tokenize(c.text)) for c in chunks]
    lengths = np.array([sum(tf.values()) for tf in tfs], dtype=np.float64)
    df: Counter = Counter()
    for tf in tfs:
        df.update(tf.keys())
    n = len(chunks)
    idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    return _BM25(tfs, lengths, float(lengths.mean()) if n else 0.0, idf)


def _bm25_scores(bm: _BM25, query: str) -> np.ndarray:
    """BM25 score of every chunk for the query."""
    scores = np.zeros(len(bm.tfs), dtype=np.float64)
    terms = set(tokenize(query))
    if not terms or bm.avgdl == 0:
        return scores
    norm = BM25_K1 * (1 - BM25_B + BM25_B * bm.lengths / bm.avgdl)
    for t in terms:
        idf = bm.idf.get(t)
        if idf is None:
            continue
        tf = np.array([d.get(t, 0) for d in bm.tfs], dtype=np.float64)
        scores += idf * tf * (BM25_K1 + 1) / (tf + norm)
    return scores


@lru_cache(maxsize=None)
def _load(path: str, mtime_ns: int) -> tuple[faiss.Index, list[Chunk], _BM25]:
    d = Path(path)
    index_file, chunks_file = d / INDEX_FILE, d / CHUNKS_FILE
    if not index_file.is_file() or not chunks_file.is_file():
        raise IndexNotFoundError(f"Index not found in '{d}'. {_HINT}")
    index = faiss.read_index(str(index_file))
    chunks = [Chunk(**c) for c in json.loads(chunks_file.read_text(encoding="utf-8"))]
    if index.ntotal == 0 or not chunks:
        raise IndexNotFoundError(f"Index in '{d}' is empty. {_HINT}")
    if index.ntotal != len(chunks):
        raise IndexNotFoundError(f"Index in '{d}' is inconsistent with chunks.json. {_HINT}")
    return index, chunks, _build_bm25(chunks)


def load_index(index_dir: Path = INDEX_DIR) -> tuple[faiss.Index, list[Chunk]]:
    """Load (cached per path and faiss.index mtime) the index and chunks; raises IndexNotFoundError."""
    path = Path(index_dir).resolve()
    try:
        mtime_ns = (path / INDEX_FILE).stat().st_mtime_ns
    except OSError:
        mtime_ns = 0  # missing; _load raises IndexNotFoundError
    index, chunks, _ = _load(str(path), mtime_ns)
    return index, chunks


def _load_all(index_dir: Path) -> tuple[faiss.Index, list[Chunk], _BM25]:
    path = Path(index_dir).resolve()
    try:
        mtime_ns = (path / INDEX_FILE).stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return _load(str(path), mtime_ns)


def clear_cache() -> None:
    """Drop cached indexes (call after re-ingesting)."""
    _load.cache_clear()


load_index.cache_clear = clear_cache  # type: ignore[attr-defined]


def _rank(question: str, vec_scores: np.ndarray, bm: _BM25, mode: str) -> list[int]:
    """Return all chunk positions ranked best first for the given mode (no k cut-off)."""
    vec_rank = [int(i) for i in np.argsort(-vec_scores, kind="stable")]
    if mode == "vector":
        return vec_rank
    kw = _bm25_scores(bm, question)
    kw_rank = [int(i) for i in np.argsort(-kw, kind="stable") if kw[i] > 0]
    if mode == "keyword":
        return kw_rank + [i for i in vec_rank if kw[i] <= 0]
    fused: dict[int, float] = {}
    for ranking in (vec_rank, kw_rank):
        for r, i in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
    vpos = {i: r for r, i in enumerate(vec_rank)}
    return sorted(fused, key=lambda i: (-fused[i], vpos[i]))


def _vector_scores(index: faiss.Index, embed_fn, question: str) -> np.ndarray:
    """Cosine similarity of the question to every chunk, in chunk order."""
    q = embed_fn([question.strip()], is_query=True).astype("float32")
    scores, ids = index.search(q, index.ntotal)
    out = np.full(index.ntotal, -np.inf, dtype=np.float64)
    for s, i in zip(scores[0], ids[0]):
        if i >= 0:
            out[i] = s
    return out


def merge_overlap(a: str, b: str, max_overlap: int = 400, min_overlap: int = 15) -> str:
    """Join two consecutive chunk texts, dropping the chunker's overlap if b starts with a's tail."""
    a, b = a.rstrip(), b.lstrip()
    for m in range(min(max_overlap, len(a), len(b)), min_overlap - 1, -1):
        if a.endswith(b[:m]):
            return a + b[m:]
    return f"{a} {b}"


def _expanded_text(chunks: list[Chunk], i: int) -> str:
    """Chunk i's text plus the next chunk's when that one is on the same source and page."""
    c = chunks[i]
    if i + 1 < len(chunks):
        nxt = chunks[i + 1]
        if nxt.chunk_id == c.chunk_id + 1 and nxt.source == c.source and nxt.page == c.page:
            return merge_overlap(c.text, nxt.text)
    return c.text


def retrieve(
    question: str, k: int = TOP_K, index_dir: Path = INDEX_DIR, embed_fn=embed_texts, mode: str = "hybrid",
    expand_neighbors: bool = False,
) -> list[RetrievedChunk]:
    """Return up to k chunks for the question, best first.

    mode "hybrid" fuses vector and BM25 rankings with Reciprocal Rank Fusion; "vector" and "keyword" use one.
    `score` is always the vector cosine similarity. With expand_neighbors, each chunk's text also includes
    the following chunk (chunk_id+1) when it is on the same source and page; ids and pages stay the original's.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}.")
    index, chunks, bm = _load_all(index_dir)
    k = min(k, index.ntotal)
    vec_scores = _vector_scores(index, embed_fn, question)
    order = _rank(question, vec_scores, bm, mode)[:k]
    logger.debug("retrieve mode=%s ids=%s", mode, order)
    return [
        RetrievedChunk(chunks[i].chunk_id, chunks[i].source, chunks[i].page,
                       _expanded_text(chunks, i) if expand_neighbors else chunks[i].text, float(vec_scores[i]))
        for i in order
    ]
