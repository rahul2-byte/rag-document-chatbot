import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from src import ingest
from src.ingest import build_index, chunk_documents, chunk_text, load_documents

DIM = 256


def fake_embed(texts, is_query=False):
    """Deterministic hashed bag-of-words, L2-normalised."""
    import zlib

    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.strip(".,").encode()) % DIM] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


def test_chunk_sizes_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(0 < len(c) <= 200 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a[-20:].split()[-1] in b  # consecutive chunks overlap
    assert chunks[-1].endswith("word399")


def test_chunk_covers_all_text_irregular_words():
    import random

    rnd = random.Random(1)
    text = " ".join("x" * rnd.randint(1, 25) for _ in range(1500))
    size, overlap = 800, 100
    chunks = chunk_text(text, size, overlap)
    covered = [False] * len(text)
    pos = 0
    prev_end = None
    for c in chunks:
        assert len(c) <= size
        i = text.find(c, pos if prev_end is None else max(0, prev_end - size))
        assert i != -1
        if prev_end is not None:
            assert i < prev_end  # consecutive chunks overlap
        for j in range(i, i + len(c)):
            covered[j] = True
        prev_end = i + len(c)
        pos = i
    assert all(covered)


def test_chunk_prefers_whitespace():
    chunks = chunk_text("abcdefghi " * 100, size=100, overlap=10)
    assert all(c.endswith("abcdefghi") for c in chunks[:-1])


def test_short_and_empty_text():
    assert chunk_text("hello   world") == ["hello world"]
    assert chunk_text("") == []
    assert chunk_text("   \n ") == []


def test_overlap_ge_size_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=20)


def make_data(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_text("Cats purr when happy.\n\nThey sleep a lot. " * 30)
    (d / "b.md").write_text("# Rockets\nRockets burn fuel to reach orbit around Earth.")
    (d / "README.md").write_text("Should be ignored")
    (d / "image.png").write_bytes(b"x")
    return d


def test_load_documents_skips_readme_and_others(tmp_path):
    pages = load_documents(make_data(tmp_path))
    assert [(s, p) for s, p, _ in pages] == [("a.txt", 1), ("b.md", 1)]
    assert "\n" not in pages[0][2]


def test_chunk_documents_keep_metadata_and_never_cross_pages():
    pages = [("x.pdf", 1, "alpha " * 300), ("x.pdf", 2, "beta " * 300), ("y.txt", 1, "gamma")]
    chunks = chunk_documents(pages)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        if c.page == 1 and c.source == "x.pdf":
            assert "beta" not in c.text
        if c.page == 2:
            assert "alpha" not in c.text and c.source == "x.pdf"
    assert chunks[-1].source == "y.txt" and chunks[-1].text == "gamma"


def test_build_index_writes_files(tmp_path):
    d, idx = make_data(tmp_path), tmp_path / "index"
    n = build_index(d, idx, embed_fn=fake_embed)
    saved = json.loads((idx / "chunks.json").read_text())
    assert n == len(saved) > 0
    index = faiss.read_index(str(idx / "faiss.index"))
    assert index.ntotal == n and index.d == DIM
    assert set(saved[0]) == {"chunk_id", "source", "page", "text"}


def test_build_index_rebuild_deletes_old(tmp_path):
    d, idx = make_data(tmp_path), tmp_path / "index"
    build_index(d, idx, embed_fn=fake_embed)
    (d / "a.txt").unlink()
    n = build_index(d, idx, embed_fn=fake_embed)
    assert n == len(json.loads((idx / "chunks.json").read_text()))


def test_build_index_no_documents(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "README.md").write_text("x")
    with pytest.raises(ValueError, match="No documents"):
        build_index(d, tmp_path / "index", embed_fn=fake_embed)


PDFS = sorted((Path(__file__).resolve().parent.parent / "data").glob("*.pdf"))


@pytest.mark.skipif(not PDFS, reason="no PDFs downloaded")
def test_load_real_pdf(tmp_path):
    (tmp_path / PDFS[0].name).write_bytes(PDFS[0].read_bytes())
    pages = ingest.load_documents(tmp_path)
    assert pages and pages[0][0] == PDFS[0].name
    assert all(p >= 1 and t for _, p, t in pages)


def test_list_documents(tmp_path):
    d = make_data(tmp_path)
    docs = ingest.list_documents(d)
    assert [x["name"] for x in docs] == ["a.txt", "b.md"]
    assert [x["type"] for x in docs] == ["txt", "md"]
    assert all(x["pages"] == 1 and x["size_bytes"] > 0 and x["modified"] > 0 for x in docs)
    assert ingest.list_documents(tmp_path / "missing") == []


def test_list_documents_unreadable_pdf(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    docs = ingest.list_documents(tmp_path)
    assert docs[0]["type"] == "pdf" and docs[0]["pages"] == 0


def test_meta_json_content(tmp_path):
    d, idx = make_data(tmp_path), tmp_path / "index"
    n = build_index(d, idx, embed_fn=fake_embed)
    meta = json.loads((idx / "meta.json").read_text())
    assert meta["embed_model"] == ingest.EMBED_MODEL
    assert meta["chunk_size"] == ingest.CHUNK_SIZE and meta["chunk_overlap"] == ingest.CHUNK_OVERLAP
    assert meta["built_at"]
    assert set(meta["documents"]) == {"a.txt", "b.md"}
    assert sum(v["chunks"] for v in meta["documents"].values()) == n
    assert set(meta["documents"]["a.txt"]) == {"pages", "chunks", "size_bytes", "modified"}


def test_index_info_fresh_then_stale(tmp_path):
    import os

    d, idx = make_data(tmp_path), tmp_path / "index"
    assert ingest.index_info(idx, d)["exists"] is False
    assert ingest.index_info(idx, d)["stale"] is True
    n = build_index(d, idx, embed_fn=fake_embed)
    info = ingest.index_info(idx, d)
    assert info["exists"] and not info["stale"]
    assert info["num_chunks"] == n and info["num_documents"] == 2
    assert info["documents"]["b.md"]["pages"] == 1 and info["built_at"]
    (d / "c.txt").write_text("new file")
    assert ingest.index_info(idx, d)["stale"] is True
    build_index(d, idx, embed_fn=fake_embed)
    assert not ingest.index_info(idx, d)["stale"]
    st = (d / "b.md").stat()
    os.utime(d / "b.md", (st.st_atime, st.st_mtime + 10))
    assert ingest.index_info(idx, d)["stale"] is True


def test_build_index_removes_old_meta(tmp_path):
    d, idx = make_data(tmp_path), tmp_path / "index"
    build_index(d, idx, embed_fn=fake_embed)
    for f in d.iterdir():
        f.unlink()
    with pytest.raises(ValueError):
        build_index(d, idx, embed_fn=fake_embed)
    assert not (idx / "meta.json").exists()
