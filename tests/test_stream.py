"""Contract v4: streaming answers, /docs, /api/followups (fake LLMs, tmp dirs, no network)."""
import json
from functools import partial
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from src import answer as A
from src.answer import REFUSAL_MESSAGE, LLMError, answer_question, answer_question_stream
from src.app import create_app
from src.intents import SUGGESTIONS
from src.qlog import QuestionLog
from src.retrieve import RetrievedChunk

TEXT = "Alpha is a thing [a.pdf, p.1].\n\n- Beta helps [b.pdf, p.2]\n- Bad one [z.pdf, p.9]"


def chunks():
    return [RetrievedChunk(1, "a.pdf", 1, "alpha text", 0.9), RetrievedChunk(2, "b.pdf", 2, "beta text", 0.8)]


def ret(q, k=4):
    return chunks()


def fake_stream(text=TEXT, size=7):
    def gen(prompt):
        for i in range(0, len(text), size):
            yield text[i:i + size]
    return gen


def no_llm(prompt):
    raise AssertionError("condense must not be called without history")


def make_client(tmp_path, *, answer_fn=None, stream_fn=None, followups_fn=None):
    answer_fn = answer_fn or partial(answer_question, retrieve_fn=ret, llm_fn=lambda p: TEXT)
    stream_fn = stream_fn or partial(answer_question_stream, retrieve_fn=ret, llm_fn=no_llm, stream_fn=fake_stream())
    kw = {"followups_fn": followups_fn} if followups_fn else {}
    app = create_app(answer_fn, tmp_path / "index", tmp_path / "data", allowed_hosts=["testserver"],
                     log_dir=tmp_path / "logs", stream_fn=stream_fn, **kw)
    return TestClient(app)


def sse(resp):
    assert resp.headers["content-type"].startswith("text/event-stream")
    blocks = [b for b in resp.text.split("\n\n") if b.strip()]
    assert all(b.startswith("data: ") and "\n" not in b for b in blocks)
    return [json.loads(b[len("data: "):]) for b in blocks]


# --------------------------------------------------------------- answer.py
def test_stream_events_and_final_equals_non_stream():
    events = list(answer_question_stream("What is alpha?", retrieve_fn=ret, llm_fn=no_llm, stream_fn=fake_stream()))
    assert events[0] == {"type": "meta", "rewritten": None}
    assert [e["type"] for e in events[1:-1]] == ["delta"] * (len(events) - 2)
    assert "".join(e["text"] for e in events[1:-1]) == TEXT
    done = events[-1]["result"]
    plain = answer_question("What is alpha?", retrieve_fn=ret, llm_fn=lambda p: TEXT)
    assert done == plain and done.invalid_citations == [("z.pdf", 9)] and "z.pdf" not in done.answer


def test_stream_passes_detail_and_k():
    seen = {}

    def r(q, k=4):
        seen["k"] = k
        return chunks()

    def s(prompt):
        seen["prompt"] = prompt
        yield "Alpha [a.pdf, p.1]."

    list(answer_question_stream("What is alpha?", retrieve_fn=r, llm_fn=no_llm, stream_fn=s, detail="brief"))
    assert seen["k"] == 4 and "concise" in seen["prompt"]
    list(answer_question_stream("What is alpha?", retrieve_fn=r, llm_fn=no_llm, stream_fn=s))
    assert seen["k"] == 8 and "thorough" in seen["prompt"]


def test_stream_error_mid_way_gives_error_done():
    def bad(prompt):
        yield "Alpha "
        raise LLMError("OpenRouter rate limit reached. Please wait and try again.")

    events = list(answer_question_stream("What is alpha?", retrieve_fn=ret, llm_fn=no_llm, stream_fn=bad))
    assert [e["type"] for e in events] == ["meta", "delta", "done"]
    assert events[-1]["result"].error.startswith("OpenRouter rate limit")


def test_stream_empty_and_refusal_and_no_index():
    ev = list(answer_question_stream("What is alpha?", retrieve_fn=ret, llm_fn=no_llm, stream_fn=lambda p: iter([])))
    assert ev[-1]["result"].error == "The model returned an empty response."
    ev = list(answer_question_stream("q?", retrieve_fn=ret, llm_fn=no_llm,
                                     stream_fn=fake_stream(REFUSAL_MESSAGE)))
    assert ev[-1]["result"].refused and ev[-1]["result"].answer == REFUSAL_MESSAGE
    ev = list(answer_question_stream("q?", retrieve_fn=lambda q, k=4: [], llm_fn=no_llm, stream_fn=no_llm))
    assert [e["type"] for e in ev] == ["done"] and ev[0]["result"].refused


# ------------------------------------------------------------------ app
def test_api_stream_matches_api_ask_and_logs_once(tmp_path):
    c = make_client(tmp_path)
    events = sse(c.post("/api/ask/stream", json={"question": "What is alpha?"}))
    assert [e["type"] for e in events][0] == "meta" and events[-1]["type"] == "done"
    assert all(e["type"] == "delta" for e in events[1:-1]) and len(events) > 3
    done = dict(events[-1])
    assert done.pop("type") == "done" and done["id"] == events[0]["id"]
    plain = c.post("/api/ask", json={"question": "What is alpha?"}).json()
    assert done.pop("id") and plain.pop("id")
    assert done == plain and done["kind"] == "answer" and [s["n"] for s in done["sources"]] == [1, 2]
    items = QuestionLog(tmp_path / "logs").read()["items"]
    assert len(items) == 2 and items[1]["id"] == events[0]["id"] and items[1]["citations"] == ["a.pdf p.1", "b.pdf p.2"]


def test_api_stream_error_and_crash(tmp_path):
    def bad(prompt):
        yield "x"
        raise LLMError("boom")

    c = make_client(tmp_path, stream_fn=partial(answer_question_stream, retrieve_fn=ret, llm_fn=no_llm, stream_fn=bad))
    events = sse(c.post("/api/ask/stream", json={"question": "What is alpha?"}))
    assert events[-1]["kind"] == "error" and events[-1]["error"] == "boom"

    def crash(q, **kw):
        yield {"type": "meta", "rewritten": None}
        raise RuntimeError("x")

    events = sse(make_client(tmp_path, stream_fn=crash).post("/api/ask/stream", json={"question": "What is alpha?"}))
    assert [e["type"] for e in events] == ["meta", "done"] and events[-1]["kind"] == "error"


@pytest.mark.parametrize("msg,kind", [("hello", "smalltalk"), ("   ", "error"), ("What is alpha? " * 200, "error")])
def test_api_stream_smalltalk_and_input_errors_are_single_done(tmp_path, msg, kind):
    def boom(q, **kw):
        raise AssertionError("must not be called")

    events = sse(make_client(tmp_path, stream_fn=boom).post("/api/ask/stream", json={"question": msg}))
    assert len(events) == 1 and events[0]["type"] == "done" and events[0]["kind"] == kind
    assert QuestionLog(tmp_path / "logs").read()["items"][0]["kind"] == kind


def test_api_detail_param_end_to_end(tmp_path):
    seen = []

    def fn(q, **kw):
        seen.append(kw.get("detail"))
        return answer_question(q, retrieve_fn=ret, llm_fn=lambda p: TEXT)

    def sfn(q, **kw):
        seen.append(kw.get("detail"))
        yield from answer_question_stream(q, retrieve_fn=ret, llm_fn=no_llm, stream_fn=fake_stream())

    c = make_client(tmp_path, answer_fn=fn, stream_fn=sfn)
    for d in ("brief", "detailed", None, "weird"):
        body = {"question": "What is alpha?"} | ({} if d is None else {"detail": d})
        c.post("/api/ask", json=body)
        c.post("/api/ask/stream", json=body)
    assert seen == ["brief"] * 2 + ["detailed"] * 6


def test_docs_route(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    (data / "notes.md").write_text("# hi")
    (data / "secret.env").write_text("KEY=1")
    (tmp_path / ".env").write_text("KEY=1")
    c = make_client(tmp_path)
    r = c.get("/docs/doc.pdf")
    assert r.status_code == 200 and r.content == b"%PDF-1.4 fake"
    assert r.headers["content-type"] == "application/pdf" and r.headers["content-disposition"].startswith("inline")
    r = c.get("/docs/notes.md")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    for bad in ("/docs/secret.env", "/docs/missing.pdf", "/docs/..%2F.env", "/docs/%2E%2E%2F.env",
                "/docs/README.md"):
        assert c.get(bad).status_code in (400, 404), bad
    assert c.get("/docs/../.env").status_code in (400, 404)


def test_followups_parsing():
    raw = ("1. What is the Govern function?\n- What does Identify cover?\n* \"How is Protect used?\"\n"
           "What are the six Functions?\nNot a question\n" + "x" * 120 + "?\n2) What does Identify cover?\n"
           "What is Recover?")
    assert A.parse_followups(raw, "What are the six functions") == [
        "What is the Govern function?", "What does Identify cover?", "How is Protect used?"]


def test_suggest_followups_prompt_and_failures():
    seen = {}

    def rec(prompt):
        seen["p"] = prompt
        return "What is A?\nWhat is B?\nWhat is C?\nWhat is D?"

    assert A.suggest_followups("Q here?", "Answer text", ["AI RMF title"], llm_fn=rec) == \
        ["What is A?", "What is B?", "What is C?"]
    assert "AI RMF title" in seen["p"] and "Answer text" in seen["p"] and "Q here?" in seen["p"]

    def boom(prompt):
        raise LLMError("x")

    assert A.suggest_followups("Q?", "A", [], llm_fn=boom) == []
    assert A.suggest_followups("", "A", [], llm_fn=rec) == []


def test_api_followups(tmp_path):
    seen = {}

    def fn(q, a, titles):
        seen.update(q=q, a=a, titles=titles)
        return ["What is B?"]

    c = make_client(tmp_path, followups_fn=fn)
    assert c.post("/api/followups", json={"question": "What is A?", "answer": "A is a."}).json() == \
        {"followups": ["What is B?"]}
    assert seen["q"] == "What is A?" and seen["titles"] == []
    assert c.post("/api/followups", json={"question": "x" * 2001, "answer": "a"}).json() == {"followups": []}
    c.post("/api/followups", json={"question": "Q?", "answer": "y" * 5000})
    assert len(seen["a"]) == 2000

    def boom(q, a, titles):
        raise RuntimeError("x")

    c = make_client(tmp_path, followups_fn=boom)
    assert c.post("/api/followups", json={"question": "Q?", "answer": "A"}).json() == {"followups": []}


@pytest.mark.parametrize("path,body", [("/api/ask/stream", {"question": "q"}),
                                       ("/api/followups", {"question": "q", "answer": "a"})])
def test_new_post_routes_are_protected(tmp_path, path, body):
    c = make_client(tmp_path, followups_fn=lambda q, a, t: [])
    assert c.post(path, json=body, headers={"Origin": "http://evil.example"}).status_code == 403
    assert c.post(path, json=body, headers={"Host": "evil.example"}).status_code == 400
    assert c.post(path, json=body, headers={"Origin": "http://testserver"}).status_code == 200
    assert c.get("/docs/x.pdf", headers={"Host": "evil.example"}).status_code == 400


# ------------------------------------------------------- SDK streaming adapters
def test_openai_stream_adapter(monkeypatch):
    import openai

    calls = {}

    class FakeStream:
        def __iter__(self):
            for piece in ("Hel", None, "lo"):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
            yield SimpleNamespace(choices=[])  # usage-only chunk

        def close(self):
            calls["closed"] = True

    class FakeClient:
        def __init__(self, **kw):
            calls["init"] = kw
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            calls["create"] = kw
            return FakeStream()

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert list(A.generate_stream("p")) == ["Hel", "lo"]
    assert calls["create"]["stream"] is True and calls["closed"]
    assert calls["init"]["base_url"] == A.OPENROUTER_BASE_URL

    def fail(self, **kw):
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(FakeClient, "create", fail)
    with pytest.raises(LLMError, match="Could not connect to the OpenRouter API"):
        list(A.generate_stream("p"))


def test_anthropic_stream_adapter(monkeypatch):
    import anthropic

    class FakeStream:
        text_stream = iter(["A", "B"])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return SimpleNamespace(stop_reason="end_turn")

    class FakeClient:
        def __init__(self, **kw):
            self.messages = SimpleNamespace(stream=lambda **kw: FakeStream())

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert list(A.generate_stream("p")) == ["A", "B"]
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        list(A.generate_stream("p"))


# ------------------------------------------------- follow-up fixes (v4.1)
def test_collapse_repeated_tags():
    f = A.collapse_repeated_tags
    assert f("Protect. [a.pdf, p.20] [a.pdf, p.20]\n- x [a.pdf, p.2][a.pdf, p.2] [a.pdf, p.2]") == \
        "Protect. [a.pdf, p.20]\n- x [a.pdf, p.2]"
    assert f("x [a.pdf, p.1] [a.pdf, p.2] y [a.pdf, p.1]") == "x [a.pdf, p.1] [a.pdf, p.2] y [a.pdf, p.1]"
    assert f("see [note] [note]") == "see [note] [note]"  # not citations: untouched
    r = A.finalize_answer("Alpha [a.pdf, p.1] [a.pdf, p.1] and [b.pdf, p.2].", chunks())
    assert r.answer == "Alpha [a.pdf, p.1] and [b.pdf, p.2]." and r.valid_citations == [("a.pdf", 1), ("b.pdf", 2)]


def test_merge_sources_same_page_without_repeating_text():
    from src.app import merge_sources
    from src.retrieve import merge_overlap

    c10, c11, c12 = ("One two three four five six seven. ", "five six seven. Eight nine ten eleven. ",
                     "ten eleven. Twelve thirteen fourteen.")
    # neighbour expansion: chunk 10's text already includes chunk 11, chunk 11's includes chunk 12
    e10, e11 = merge_overlap(c10, c11, min_overlap=5), merge_overlap(c11, c12, min_overlap=5)
    cited = [RetrievedChunk(11, "a.pdf", 20, e11, 0.5), RetrievedChunk(3, "b.pdf", 1, "Bee.", 0.9),
             RetrievedChunk(10, "a.pdf", 20, e10, 0.7)]
    out = merge_sources(cited)
    assert [(s["n"], s["source"], s["page"], s["score"]) for s in out] == [(1, "a.pdf", 20, 0.7), (2, "b.pdf", 1, 0.9)]
    assert out[0]["text"] == "One two three four five six seven. Eight nine ten eleven. Twelve thirteen fourteen."
    gap = merge_sources([RetrievedChunk(1, "a.pdf", 2, "First part.", 0.1),
                         RetrievedChunk(5, "a.pdf", 2, "Separate part.", 0.2)])
    assert gap[0]["text"] == "First part. Separate part."
    dup = merge_sources([RetrievedChunk(1, "a.pdf", 2, "Same text.", 0.1), RetrievedChunk(2, "a.pdf", 2, "Same text.", 0.1)])
    assert len(dup) == 1 and dup[0]["text"] == "Same text."


def test_merged_sources_in_ask_and_stream_done(tmp_path):
    same_page = [RetrievedChunk(1, "a.pdf", 1, "alpha text", 0.9), RetrievedChunk(2, "a.pdf", 1, "alpha more", 0.7)]
    r = lambda q, k=4: same_page  # noqa: E731
    text = "Alpha [a.pdf, p.1] [a.pdf, p.1]."
    c = make_client(tmp_path, answer_fn=partial(answer_question, retrieve_fn=r, llm_fn=lambda p: text),
                    stream_fn=partial(answer_question_stream, retrieve_fn=r, llm_fn=no_llm, stream_fn=fake_stream(text)))
    plain = c.post("/api/ask", json={"question": "What is alpha?"}).json()
    done = sse(c.post("/api/ask/stream", json={"question": "What is alpha?"}))[-1]
    for body in (plain, done):
        assert body["answer"] == "Alpha [a.pdf, p.1]."
        assert body["sources"] == [{"n": 1, "source": "a.pdf", "page": 1, "score": 0.9, "text": "alpha text alpha more"}]


@pytest.mark.parametrize("env,expected", [(None, {"reasoning": {"enabled": False}}), ("off", {"reasoning": {"enabled": False}}),
                                          ("on", None)])
def test_openrouter_reasoning_toggle(monkeypatch, env, expected):
    import openai

    seen = []

    class FakeStream(list):
        def close(self):
            pass

    class FakeClient:
        def __init__(self, **kw):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            seen.append(kw.get("extra_body"))
            if kw.get("stream"):
                chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))])
                return FakeStream([chunk])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Hi"))])

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    if env is None:
        monkeypatch.delenv("LLM_REASONING", raising=False)
    else:
        monkeypatch.setenv("LLM_REASONING", env)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert A.generate("p") == "Hi"
    assert list(A.generate_stream("p")) == ["Hi"]
    assert seen == [expected, expected]
    monkeypatch.setenv("LLM_PROVIDER", "openai")  # plain OpenAI never gets the OpenRouter field
    A.generate("p")
    assert seen[-1] is None
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_REASONING", "maybe")
    with pytest.raises(LLMError, match="LLM_REASONING"):
        A.generate("p")
