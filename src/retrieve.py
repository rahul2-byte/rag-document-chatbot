"""STUB – contract only. Agent B replaces this file. See TASKS.md."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.ingest import INDEX_DIR, embed_texts

TOP_K = 4


class IndexNotFoundError(RuntimeError):
    """Index missing or empty."""


@dataclass
class RetrievedChunk:
    chunk_id: int
    source: str
    page: int
    text: str
    score: float


def load_index(index_dir: Path = INDEX_DIR):
    raise NotImplementedError


def retrieve(question: str, k: int = TOP_K, index_dir: Path = INDEX_DIR, embed_fn=embed_texts) -> list[RetrievedChunk]:
    raise NotImplementedError
