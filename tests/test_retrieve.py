import pytest

from src.ingest import build_index
from src.retrieve import IndexNotFoundError, load_index, retrieve
from tests.test_ingest import fake_embed


@pytest.fixture
def idx(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "cats.txt").write_text("Cats purr when they are happy and sleep many hours.")
    (d / "space.md").write_text("Rockets burn fuel to reach orbit around the Earth.")
    (d / "cars.txt").write_text("Electric cars use batteries and electric motors.")
    out = tmp_path / "index"
    build_index(d, out, embed_fn=fake_embed)
    load_index.cache_clear()
    yield out
    load_index.cache_clear()


def test_relevant_first_with_metadata(idx):
    res = retrieve("Why do rockets need fuel to reach orbit?", k=3, index_dir=idx, embed_fn=fake_embed)
    assert res[0].source == "space.md" and res[0].page == 1
    assert "Rockets" in res[0].text
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)


def test_k_behaviour(idx):
    assert len(retrieve("cats", k=1, index_dir=idx, embed_fn=fake_embed)) == 1
    assert len(retrieve("cats", k=50, index_dir=idx, embed_fn=fake_embed)) == 3


def test_missing_index(tmp_path):
    with pytest.raises(IndexNotFoundError, match="src.ingest"):
        retrieve("hello", index_dir=tmp_path / "nope", embed_fn=fake_embed)


def test_empty_index(tmp_path):
    import faiss

    (tmp_path / "chunks.json").write_text("[]")
    faiss.write_index(faiss.IndexFlatIP(8), str(tmp_path / "faiss.index"))
    load_index.cache_clear()
    with pytest.raises(IndexNotFoundError):
        load_index(tmp_path)


@pytest.mark.parametrize("q", ["", "   \n"])
def test_empty_question(idx, q):
    with pytest.raises(ValueError):
        retrieve(q, index_dir=idx, embed_fn=fake_embed)


def test_query_flag_passed(idx):
    seen = {}

    def spy(texts, is_query=False):
        seen["q"] = is_query
        return fake_embed(texts, is_query)

    retrieve("cats", index_dir=idx, embed_fn=spy)
    assert seen["q"] is True


def test_auto_reload_after_rebuild(tmp_path):
    import os

    d = tmp_path / "data"
    d.mkdir()
    (d / "one.txt").write_text("Cats purr when they are happy.")
    idx = tmp_path / "index"
    build_index(d, idx, embed_fn=fake_embed)
    load_index.cache_clear()
    assert retrieve("cats", k=1, index_dir=idx, embed_fn=fake_embed)[0].source == "one.txt"
    (d / "one.txt").unlink()
    (d / "two.txt").write_text("Rockets burn fuel to reach orbit.")
    build_index(d, idx, embed_fn=fake_embed)
    st = (idx / "faiss.index").stat()
    os.utime(idx / "faiss.index", ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    res = retrieve("rockets", k=1, index_dir=idx, embed_fn=fake_embed)
    assert res[0].source == "two.txt"
    load_index.cache_clear()


def test_tokenizer_keeps_hyphenated_and_drops_stopwords():
    from src.retrieve import tokenize

    assert tokenize("What are the Protect-P and Tier-1 functions of NIST?") == [
        "protect-p", "tier-1", "functions", "nist"
    ]


def test_keyword_only_match_returned_in_hybrid(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for n in "abcde":
        (d / f"{n}.txt").write_text(f"alpha beta gamma delta epsilon {n}x")
    (d / "z.txt").write_text("theta iota kappa lambda mu nu xi zorgblat")
    out = tmp_path / "index"
    build_index(d, out, embed_fn=fake_embed)
    load_index.cache_clear()
    q = "zorgblat alpha beta gamma delta epsilon"
    vec = retrieve(q, k=3, index_dir=out, embed_fn=fake_embed, mode="vector")
    assert "z.txt" not in [r.source for r in vec]
    hyb = retrieve(q, k=3, index_dir=out, embed_fn=fake_embed)
    assert "z.txt" in [r.source for r in hyb]
    load_index.cache_clear()


def test_hybrid_respects_k_and_score_is_cosine(idx):
    q = "Why do rockets need fuel to reach orbit?"
    hyb = retrieve(q, k=2, index_dir=idx, embed_fn=fake_embed)
    assert len(hyb) == 2
    vec = {r.chunk_id: r.score for r in retrieve(q, k=3, index_dir=idx, embed_fn=fake_embed, mode="vector")}
    for r in hyb:
        assert r.score == pytest.approx(vec[r.chunk_id])


def test_invalid_mode(idx):
    with pytest.raises(ValueError):
        retrieve("cats", index_dir=idx, embed_fn=fake_embed, mode="nope")


def test_merge_overlap():
    from src.retrieve import merge_overlap

    assert merge_overlap("one two three four five six", "four five six seven eight", min_overlap=5) == \
        "one two three four five six seven eight"
    assert merge_overlap("alpha", "beta") == "alpha beta"


def test_expanded_text_same_source_and_page_only():
    from src.ingest import Chunk
    from src.retrieve import _expanded_text

    cs = [Chunk(0, "a.pdf", 1, "first"), Chunk(1, "a.pdf", 1, "second"), Chunk(2, "a.pdf", 2, "third"),
          Chunk(3, "b.pdf", 2, "fourth")]
    assert _expanded_text(cs, 0) == "first second"
    assert _expanded_text(cs, 1) == "second"  # next chunk is on another page
    assert _expanded_text(cs, 2) == "third"  # next chunk is another source
    assert _expanded_text(cs, 3) == "fourth"  # last chunk


def test_retrieve_expand_neighbors(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    words = " ".join(f"w{i}" for i in range(400))
    (d / "long.txt").write_text("Tiers are listed here. " + words + " Adaptive is Tier four.")
    out = tmp_path / "index"
    build_index(d, out, embed_fn=fake_embed)
    load_index.cache_clear()
    _, chunks = load_index(out)
    assert len(chunks) >= 2
    plain = retrieve("Tiers are listed here", k=1, index_dir=out, embed_fn=fake_embed, mode="keyword")
    exp = retrieve("Tiers are listed here", k=1, index_dir=out, embed_fn=fake_embed, mode="keyword",
                   expand_neighbors=True)
    assert plain[0].chunk_id == exp[0].chunk_id == 0 and exp[0].page == plain[0].page
    assert plain[0].text == chunks[0].text
    assert exp[0].text.startswith(chunks[0].text) and exp[0].text.endswith(chunks[1].text)
    assert len(exp[0].text) < len(chunks[0].text) + len(chunks[1].text)  # overlap removed
    load_index.cache_clear()
