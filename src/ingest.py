"""Load documents, chunk them, embed them and build the FAISS index."""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"
SUPPORTED_EXTS = (".pdf", ".txt", ".md")


@dataclass
class Chunk:
    chunk_id: int
    source: str  # file name only
    page: int  # 1-based PDF page; 1 for .txt/.md
    text: str


def _normalise(text: str) -> str:
    """Collapse all whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _data_files(data_dir: Path) -> list[Path]:
    """Sorted supported files in data_dir, excluding README.md."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    return [
        p for p in sorted(data_dir.iterdir())
        if p.is_file() and p.name.lower() != "readme.md" and p.suffix.lower() in SUPPORTED_EXTS
    ]


def list_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Describe each supported document: name, type, size_bytes, pages, modified."""
    docs: list[dict] = []
    for path in _data_files(data_dir):
        st = path.stat()
        kind = path.suffix.lower().lstrip(".")
        pages = 1
        if kind == "pdf":
            try:
                from pypdf import PdfReader

                pages = len(PdfReader(str(path)).pages)
            except Exception:  # noqa: BLE001 - unreadable PDF
                logger.warning("Could not read %s", path.name)
                pages = 0
        docs.append(
            {"name": path.name, "type": kind, "size_bytes": st.st_size, "pages": pages, "modified": st.st_mtime}
        )
    return docs


def load_documents(data_dir: Path) -> list[tuple[str, int, str]]:
    """Return (source, page, text) per page for .pdf/.txt/.md files, skipping README.md."""
    data_dir = Path(data_dir)
    pages: list[tuple[str, int, str]] = []
    for path in _data_files(data_dir):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, start=1):
                text = _normalise(page.extract_text() or "")
                if text:
                    pages.append((path.name, i, text))
        else:
            text = _normalise(path.read_text(encoding="utf-8", errors="replace"))
            if text:
                pages.append((path.name, 1, text))
        logger.info("Loaded %s", path.name)
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into ~size-char chunks with overlap, ending at whitespace when possible."""
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
    text = _normalise(text)
    n = len(text)
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Prefer to break at whitespace in the last 20% of the window.
            ws = text.rfind(" ", start + int(size * 0.8), end)
            if ws > start:
                end = ws
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        # Next start is derived from the actual end so no text is dropped.
        nxt = max(end - overlap, start + 1)
        if text[nxt - 1] != " ":  # start at a word boundary if still inside the overlap
            sp = text.find(" ", nxt, end)
            if sp != -1 and sp + 1 < end:
                nxt = sp + 1
        start = nxt
    return chunks


def chunk_documents(pages: list[tuple[str, int, str]]) -> list[Chunk]:
    """Chunk each page separately; chunk_ids are sequential."""
    chunks: list[Chunk] = []
    for source, page, text in pages:
        for piece in chunk_text(text):
            chunks.append(Chunk(len(chunks), source, page, piece))
    return chunks


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Embed texts to float32 L2-normalised vectors; queries get QUERY_PREFIX."""
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    vecs = _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.ascontiguousarray(vecs, dtype=np.float32)


def build_index(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR, embed_fn=embed_texts) -> int:
    """Rebuild the index from scratch; return the number of chunks."""
    import faiss

    data_dir, index_dir = Path(data_dir), Path(index_dir)
    (index_dir / META_FILE).unlink(missing_ok=True)  # stale meta must never outlive a rebuild attempt
    pages = load_documents(data_dir)
    if not pages:
        raise ValueError(f"No documents with text found in '{data_dir}' (.pdf, .txt, .md).")
    chunks = chunk_documents(pages)
    if not chunks:
        raise ValueError(f"No chunks produced from documents in '{data_dir}'.")
    logger.info("%d pages -> %d chunks", len(pages), len(chunks))

    index_dir.mkdir(parents=True, exist_ok=True)
    for name in (INDEX_FILE, CHUNKS_FILE, META_FILE):
        (index_dir / name).unlink(missing_ok=True)

    vecs = np.ascontiguousarray(embed_fn([c.text for c in chunks]), dtype=np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(index_dir / INDEX_FILE))
    (index_dir / CHUNKS_FILE).write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False), encoding="utf-8"
    )
    per_doc: dict[str, dict] = {}
    for d in list_documents(data_dir):
        per_doc[d["name"]] = {
            "pages": d["pages"],
            "chunks": sum(1 for c in chunks if c.source == d["name"]),
            "size_bytes": d["size_bytes"],
            "modified": d["modified"],
        }
    meta = {
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "documents": per_doc,
    }
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    logger.info("Wrote index with %d vectors to %s", index.ntotal, index_dir)
    return len(chunks)


def index_info(index_dir: Path = INDEX_DIR, data_dir: Path = DATA_DIR) -> dict:
    """Summarise the built index and whether it is stale relative to data_dir."""
    index_dir = Path(index_dir)
    info: dict = {
        "exists": (index_dir / INDEX_FILE).is_file() and (index_dir / CHUNKS_FILE).is_file(),
        "built_at": None, "num_chunks": 0, "num_documents": 0,
        "documents": {}, "embed_model": None, "stale": True,
    }
    meta_path = index_dir / META_FILE
    meta = None
    if info["exists"] and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
    if info["exists"]:
        try:
            info["num_chunks"] = len(json.loads((index_dir / CHUNKS_FILE).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    if meta is None:
        return info  # missing index or no/bad meta -> stale
    docs = meta.get("documents", {})
    info.update(
        built_at=meta.get("built_at"),
        embed_model=meta.get("embed_model"),
        documents={n: {"pages": d.get("pages", 0), "chunks": d.get("chunks", 0)} for n, d in docs.items()},
        num_documents=len(docs),
    )
    current = {p.name: (p.stat().st_size, p.stat().st_mtime) for p in _data_files(data_dir)}
    recorded = {n: (d.get("size_bytes"), d.get("modified")) for n, d in docs.items()}
    info["stale"] = current != recorded
    return info


def main() -> int:
    """CLI entry: rebuild the index."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        n = build_index()
    except Exception as exc:  # noqa: BLE001 - report any failure clearly
        print(f"Ingest failed: {exc}", file=sys.stderr)
        return 1
    print(f"Done: indexed {n} chunks into {INDEX_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
