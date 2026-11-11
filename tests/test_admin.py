"""Tests for the owner console (FastAPI TestClient, fake embeddings)."""
import threading

import pytest
from fastapi.testclient import TestClient

from src import admin
from src.retrieve import clear_cache
from tests.test_ingest import fake_embed


@pytest.fixture
def env(tmp_path):
    clear_cache()
    data, index = tmp_path / "data", tmp_path / "index"
    data.mkdir()
    client = TestClient(admin.create_admin_app(data, index, fake_embed, allowed_hosts=["testserver"]))
    return client, data, index


def up(client, *files):
    return client.post("/api/upload", files=[("files", f) for f in files])


def test_pages_served(env):
    c, _, _ = env
    assert "Owner console" in c.get("/").text
    for p in ("/static/admin.js", "/static/admin.css", "/shared/base.css"):
        assert c.get(p).status_code == 200


def test_upload_good_files(env):
    c, data, _ = env
    r = up(c, ("a.txt", b"hello cats", "text/plain"), ("b.md", b"# dogs", "text/markdown")).json()
    assert [s["name"] for s in r["saved"]] == ["a.txt", "b.md"] and r["errors"] == []
    assert (data / "a.txt").read_bytes() == b"hello cats"
    again = up(c, ("a.txt", b"new", "text/plain")).json()
    assert again["saved"][0]["replaced"] is True


def test_upload_rejections(env):
    c, data, _ = env
    r = up(c, ("x.exe", b"abc", "a/b"), ("empty.txt", b"", "text/plain"),
           ("README.md", b"hi", "text/plain"), ("bad.pdf", b"not a pdf", "application/pdf")).json()
    assert r["saved"] == [] and len(r["errors"]) == 4
    msgs = {e["name"]: e["message"] for e in r["errors"]}
    assert "Unsupported" in msgs["x.exe"] and "empty" in msgs["empty.txt"]
    assert "reserved" in msgs["README.md"] and "PDF" in msgs["bad.pdf"]
    assert list(data.iterdir()) == []


def test_upload_traversal_name_is_sanitised(env):
    c, data, _ = env
    r = up(c, ("../../evil.txt", b"data", "text/plain")).json()
    assert r["saved"][0]["name"] == "evil.txt"
    assert (data / "evil.txt").exists() and not (data.parent.parent / "evil.txt").exists()


def test_upload_too_large(env, monkeypatch):
    c, data, _ = env
    monkeypatch.setattr(admin, "MAX_UPLOAD_BYTES", 10)
    r = up(c, ("big.txt", b"x" * 50, "text/plain")).json()
    assert "50 MB" in r["errors"][0]["message"] and not (data / "big.txt").exists()


def test_upload_no_files(env):
    c, _, _ = env
    assert up(c).json()["errors"]


def test_delete(env):
    c, data, _ = env
    (data / "a.txt").write_text("x")
    assert c.delete("/api/documents/a.txt").json()["ok"] is True
    assert not (data / "a.txt").exists()
    assert c.delete("/api/documents/nope.txt").status_code == 404
    (data.parent / "secret.txt").write_text("s")
    assert c.delete("/api/documents/..%2Fsecret.txt").status_code == 404
    assert (data.parent / "secret.txt").exists()


def test_rebuild_status_and_preview(env):
    c, data, _ = env
    (data / "cats.txt").write_text("Cats purr when they are happy. " * 20)
    (data / "dogs.txt").write_text("Dogs bark at strangers loudly. " * 20)
    s = c.get("/api/status").json()
    assert s["exists"] is False and s["stale"] is True and len(s["documents"]) == 2
    assert all(d["in_index"] is False for d in s["documents"])
    r = c.post("/api/rebuild")
    body = r.json()
    assert r.status_code == 200 and body["ok"] and body["num_chunks"] >= 2 and "Indexed" in body["message"]
    s = c.get("/api/status").json()
    assert s["exists"] and not s["stale"] and s["num_documents"] == 2
    assert all(d["in_index"] and d["chunks"] >= 1 for d in s["documents"])
    res = c.post("/api/preview", json={"question": "why do cats purr"}).json()["results"]
    assert res and res[0]["source"] == "cats.txt"
    assert {"n", "source", "page", "score", "text"} <= set(res[0])


def test_preview_without_index_and_empty(env):
    c, _, _ = env
    assert c.post("/api/preview", json={"question": "x"}).json()["results"] == []
    assert "first" in c.post("/api/preview", json={"question": " "}).json()["message"]


def test_rebuild_conflict_409(tmp_path):
    clear_cache()
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("cats " * 50)
    started, release = threading.Event(), threading.Event()

    def slow_embed(texts, is_query=False):
        started.set()
        release.wait(5)
        return fake_embed(texts, is_query)

    c = TestClient(admin.create_admin_app(data, tmp_path / "index", slow_embed, allowed_hosts=["testserver"]))
    out = {}
    t = threading.Thread(target=lambda: out.update(r=c.post("/api/rebuild")))
    t.start()
    assert started.wait(5)
    try:
        assert c.post("/api/rebuild").status_code == 409
        assert c.post("/api/upload", files=[("files", ("n.txt", b"x", "text/plain"))]).status_code == 409
        assert c.delete("/api/documents/a.txt").status_code == 409
    finally:
        release.set()
        t.join(10)
    assert out["r"].status_code == 200


def test_foreign_host_rejected(env):
    c, _, _ = env
    assert c.get("/api/status", headers={"Host": "evil.example"}).status_code == 400


def test_cross_origin_writes_rejected(env):
    c, data, _ = env
    bad = {"Origin": "http://evil.example"}
    assert c.post("/api/upload", files=[("files", ("a.txt", b"x", "text/plain"))], headers=bad).status_code == 403
    assert c.post("/api/rebuild", headers=bad).status_code == 403
    assert c.post("/api/preview", json={"question": "q"}, headers=bad).status_code == 403
    assert not (data / "a.txt").exists()
    assert c.post("/api/preview", json={"question": "q"}, headers={"Origin": "http://testserver"}).status_code == 200


def test_questions_endpoint(tmp_path):
    from src.qlog import QuestionLog

    log = QuestionLog(tmp_path / "logs")
    a = log.log_question("first", "answer", 10, citations=["a.pdf p.1"])
    log.log_question("hello", "smalltalk", 1)
    log.add_feedback(a, "down")
    c = TestClient(admin.create_admin_app(tmp_path / "d", tmp_path / "i", fake_embed, allowed_hosts=["testserver"],
                                          log_dir=tmp_path / "logs"))
    body = c.get("/api/questions").json()
    assert body["stats"]["total"] == 2 and body["stats"]["down"] == 1 and body["stats"]["smalltalk"] == 1
    assert [i["question"] for i in body["items"]] == ["hello", "first"]
    assert body["items"][1]["feedback"] == "down" and body["items"][1]["citations"] == ["a.pdf p.1"]
    assert len(c.get("/api/questions?limit=1").json()["items"]) == 1
    assert c.get("/api/questions?limit=0").status_code == 422
    empty = TestClient(admin.create_admin_app(tmp_path / "d", tmp_path / "i", fake_embed,
                                              allowed_hosts=["testserver"], log_dir=tmp_path / "none"))
    assert empty.get("/api/questions").json()["items"] == []
