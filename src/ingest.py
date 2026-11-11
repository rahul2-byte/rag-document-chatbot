"""STUB – contract only. Agent B replaces this file. See TASKS.md."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Chunk:
    chunk_id: int
    source: str
    page: int
    text: str


def load_documents(data_dir: Path) -> list[tuple[str, int, str]]:
    raise NotImplementedError


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    raise NotImplementedError


def chunk_documents(pages: list[tuple[str, int, str]]) -> list[Chunk]:
    raise NotImplementedError


def embed_texts(texts: list[str], is_query: bool = False):
    raise NotImplementedError


def build_index(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR, embed_fn=embed_texts) -> int:
    raise NotImplementedError
