"""Tests for src.answer using fake retrieve/LLM functions (no network)."""
import pytest

from src import answer as A
from src.answer import REFUSAL_MESSAGE, LLMError, answer_question
from src.retrieve import IndexNotFoundError, RetrievedChunk


def chunks():
    return [
        RetrievedChunk(1, "a.pdf", 1, "alpha text", 0.9),
        RetrievedChunk(2, "b.pdf", 2, "beta text", 0.8),
        RetrievedChunk(3, "a.pdf", 1, "alpha more", 0.7),
    ]


def ret(q, k=4):
    return chunks()


def llm(text):
    return lambda prompt: text


def test_extract_variants():
    t = "x [a.pdf, p.1] y [b.pdf, p. 2] z [c.txt, p 3] [d.md, page 4] [e.pdf, p.5; f.pdf, p.6]"
    assert A.extract_citations(t) == [
        ("a.pdf", 1), ("b.pdf", 2), ("c.txt", 3), ("d.md", 4), ("e.pdf", 5), ("f.pdf", 6),
    ]


def test_extract_ignores_non_citations():
    assert A.extract_citations("see [1] and [note]") == []


def test_validate_split_dedup():
    valid, invalid = A.validate_citations(
        "[a.pdf, p.1] [b.pdf, p.9] [a.pdf, p.1] [b.pdf, p.9] [b.pdf, p.2]", chunks()
    )
    assert valid == [("a.pdf", 1), ("b.pdf", 2)]
    assert invalid == [("b.pdf", 9)]


def test_valid_answer():
    r = answer_question("q?", retrieve_fn=ret, llm_fn=llm("Alpha [a.pdf, p.1]."))
    assert not r.refused and r.error is None
    assert [c.chunk_id for c in r.cited_chunks] == [1, 3]
    assert r.valid_citations == [("a.pdf", 1)]


def test_invalid_tag_removed_and_note():
    r = answer_question("q?", retrieve_fn=ret, llm_fn=llm("Alpha [a.pdf, p.1] beta [z.pdf, p.7]."))
    assert "z.pdf" not in r.answer
    assert "Note: removed 1 citation(s) that did not match the retrieved passages." in r.answer
    assert r.invalid_citations == [("z.pdf", 7)]
    assert "[a.pdf, p.1]" in r.answer


def test_uncited_answer_rejected():
    r = answer_question("q?", retrieve_fn=ret, llm_fn=llm("Some unsupported claim."))
    assert r.refused and r.answer == REFUSAL_MESSAGE and r.cited_chunks == []


def test_only_invalid_citations_rejected():
    r = answer_question("q?", retrieve_fn=ret, llm_fn=llm("Claim [z.pdf, p.7]."))
    assert r.refused and r.answer == REFUSAL_MESSAGE


def test_refusal_detected():
    assert A.is_refusal(REFUSAL_MESSAGE)
    assert A.is_refusal("Sorry, I Cannot Find that.")
    assert not A.is_refusal("Alpha [a.pdf, p.1]")
    r = answer_question("q?", retrieve_fn=ret, llm_fn=llm(REFUSAL_MESSAGE))
    assert r.refused and r.answer == REFUSAL_MESSAGE and r.cited_chunks == [] and r.error is None


def test_empty_question():
    def boom(*a, **k):
        raise AssertionError("must not be called")

    r = answer_question("   ", retrieve_fn=boom, llm_fn=boom)
    assert r.answer == "Please enter a question." and r.error


def test_index_not_found():
    def bad(q, k=4):
        raise IndexNotFoundError("Run python -m src.ingest")

    r = answer_question("q?", retrieve_fn=bad, llm_fn=llm("x"))
    assert r.answer == "Run python -m src.ingest" and r.error and not r.refused


def test_llm_error():
    def bad(p):
        raise LLMError("no key")

    r = answer_question("q?", retrieve_fn=ret, llm_fn=bad)
    assert r.answer == "no key" and r.error == "no key" and not r.refused


def test_unexpected_exception():
    def bad(p):
        raise ValueError("kaboom")

    r = answer_question("q?", retrieve_fn=ret, llm_fn=bad)
    assert r.error and r.answer.startswith("Something went wrong") and "kaboom" not in r.answer


def test_empty_retrieval_no_llm_call():
    def boom(p):
        raise AssertionError("must not be called")

    r = answer_question("q?", retrieve_fn=lambda q, k=4: [], llm_fn=boom)
    assert r.refused and r.answer == REFUSAL_MESSAGE


def test_prompt_contains_tags_and_refusal():
    p = A.build_prompt("What?", chunks())
    assert "[a.pdf, p.1]" in p and "[b.pdf, p.2]" in p and REFUSAL_MESSAGE in p and "What?" in p


def test_generate_missing_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        A.generate("hi")


def test_generate_missing_openai_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "OpenAI")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        A.generate("hi")


def test_generate_missing_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(LLMError, match="LLM_MODEL"):
        A.generate("hi")


def test_generate_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "foo")
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(LLMError, match="LLM_PROVIDER"):
        A.generate("hi")


def test_generate_anthropic_max_tokens_no_text(monkeypatch):
    import sys
    import types

    class _Err(Exception):
        pass

    resp = types.SimpleNamespace(stop_reason="max_tokens", content=[types.SimpleNamespace(type="thinking")])
    fake = types.ModuleType("anthropic")
    fake.AuthenticationError = fake.RateLimitError = fake.APIStatusError = fake.APIConnectionError = _Err
    fake.Anthropic = lambda **kw: types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kw: resp)
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(LLMError, match="ran out of output tokens"):
        A.generate("hi")


def test_generate_missing_openrouter_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "google/gemini-2.5-flash-lite")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        A.generate("hi")


def test_generate_openrouter_passes_base_url_and_key(monkeypatch):
    import sys
    import types

    seen = {}
    msg = types.SimpleNamespace(content="  hello [a.pdf, p.1]  ")
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    def create(**kw):
        seen["create"] = kw
        return resp

    def fake_openai(**kw):
        seen["init"] = kw
        return types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )

    fake = types.ModuleType("openai")
    fake.AuthenticationError = fake.RateLimitError = fake.APIStatusError = fake.APIConnectionError = Exception
    fake.OpenAI = fake_openai
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setenv("LLM_PROVIDER", "OpenRouter")
    monkeypatch.setenv("LLM_MODEL", "google/gemini-2.5-flash-lite")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert A.generate("hi") == "hello [a.pdf, p.1]"
    assert seen["init"]["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["init"]["api_key"] == "or-key"
    assert seen["create"]["model"] == "google/gemini-2.5-flash-lite"


# ------------------------------------------------------- follow-ups (v3)
class FakeLLM:
    """Returns `condensed` for condense prompts and `answer` for answer prompts; records prompts."""

    def __init__(self, condensed="What is alpha in a.pdf?", answer="Alpha [a.pdf, p.1]."):
        self.condensed, self.answer, self.prompts = condensed, answer, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if "Standalone question:" in prompt:
            if isinstance(self.condensed, Exception):
                raise self.condensed
            return self.condensed
        return self.answer


HIST = [{"role": "user", "content": "What is alpha?"}, {"role": "assistant", "content": "Alpha is X [a.pdf, p.1]."}]


def test_no_history_no_condense_call():
    llm_ = FakeLLM()
    r = answer_question("What is alpha?", retrieve_fn=ret, llm_fn=llm_)
    assert len(llm_.prompts) == 1 and r.rewritten is None and not r.refused


def test_assistant_only_history_no_condense():
    llm_ = FakeLLM()
    answer_question("q?", retrieve_fn=ret, llm_fn=llm_, history=[{"role": "assistant", "content": "hi"}])
    assert len(llm_.prompts) == 1


def test_follow_up_condensed_and_used_for_retrieval_and_prompt():
    seen = {}

    def r(q, k=4):
        seen["q"] = q
        return chunks()

    llm_ = FakeLLM(condensed='"Tell me more about alpha in a.pdf"')
    res = answer_question("tell me more about that", retrieve_fn=r, llm_fn=llm_, history=HIST)
    assert len(llm_.prompts) == 2
    assert "What is alpha?" in llm_.prompts[0] and "tell me more about that" in llm_.prompts[0]
    assert seen["q"] == "Tell me more about alpha in a.pdf"
    assert res.rewritten == "Tell me more about alpha in a.pdf"
    assert "Question: Tell me more about alpha in a.pdf" in llm_.prompts[1]


def test_condense_unchanged_gives_no_rewritten():
    llm_ = FakeLLM(condensed="What is beta?")
    res = answer_question("What is beta?", retrieve_fn=ret, llm_fn=llm_, history=HIST)
    assert res.rewritten is None and len(llm_.prompts) == 2


@pytest.mark.parametrize("bad", ["", "   ", "x" * 1000, REFUSAL_MESSAGE, LLMError("down"), ValueError("boom")])
def test_condense_falls_back_to_original(bad):
    seen = {}

    def r(q, k=4):
        seen["q"] = q
        return chunks()

    res = answer_question("tell me more", retrieve_fn=r, llm_fn=FakeLLM(condensed=bad), history=HIST)
    assert seen["q"] == "tell me more" and res.rewritten is None and not res.error


def test_clean_history_limits():
    h = [{"role": "user", "content": "x" * 3000}] + [{"role": "assistant", "content": f"m{i}"} for i in range(10)]
    h += [{"role": "system", "content": "evil"}, "junk", {"role": "user", "content": 5}]
    out = A.clean_history(h)
    assert len(out) == 6 and all(m["role"] in ("user", "assistant") for m in out)
    assert len(A.clean_history([{"role": "user", "content": "y" * 3000}])[0]["content"]) == 2000
    assert A.clean_history(None) == []


@pytest.mark.parametrize("q,k_brief,k_detailed", [
    ("Compare the AI RMF and the Privacy Framework", 6, 8), ("What is the difference between A and B?", 6, 8),
    ("AI RMF vs CSF", 6, 8), ("Do both frameworks mention risk?", 6, 8), ("What is alpha?", 4, 8),
])
def test_comparison_uses_more_passages(q, k_brief, k_detailed):
    for detail, k in (("brief", k_brief), ("detailed", k_detailed), (None, k_detailed), ("junk", k_detailed)):
        seen = {}

        def r(question, k=4):
            seen["k"] = max(seen.get("k", 0), k)
            return chunks()

        kw = {} if detail is None else {"detail": detail}
        answer_question(q, retrieve_fn=r, llm_fn=llm("Alpha [a.pdf, p.1]."), **kw)
        assert seen["k"] == k, (detail, q)


@pytest.mark.parametrize("q,sides", [
    ("Compare the AI RMF and the Privacy Framework", ["the AI RMF", "the Privacy Framework"]),
    ("What is the difference between the CSF and the Privacy Framework?", ["the CSF", "the Privacy Framework"]),
    ("How does the AI RMF compare with the CSF 2.0?", ["the AI RMF", "the CSF 2.0"]),
    ("AI RMF vs CSF", ["AI RMF", "CSF"]),
    ("Do both frameworks mention risk?", []),
])
def test_comparison_sides(q, sides):
    assert A.comparison_sides(q) == sides


def test_comparison_retrieval_covers_both_sides():
    pool = {
        "the AI RMF": [RetrievedChunk(i, "ai.pdf", i, "ai", 0.9) for i in range(1, 4)],
        "the Privacy Framework": [RetrievedChunk(i, "priv.pdf", i, "p", 0.9) for i in range(11, 14)],
    }
    full = [RetrievedChunk(i, "ai.pdf", i, "ai", 0.9) for i in (1, 2, 21, 22, 23, 24)]
    calls = []

    def r(q, k=4):
        calls.append((q, k))
        return pool.get(q, full)[:k]

    got = A.retrieve_comparison("Compare the AI RMF and the Privacy Framework", r)
    ids = [c.chunk_id for c in got]
    assert len(ids) == 6 and len(set(ids)) == 6
    assert {c.source for c in got} == {"ai.pdf", "priv.pdf"} and ids[:2] == [1, 11]
    assert ("Compare the AI RMF and the Privacy Framework", 6) in calls and ("the AI RMF", 3) in calls


def test_default_retriever_expands_neighbors(monkeypatch):
    seen = {}

    def fake_retrieve(q, k=4, **kw):
        seen.update(kw)
        return chunks()

    monkeypatch.setattr(A, "retrieve", fake_retrieve)
    A.retrieve_expanded("q", k=4)
    assert seen == {"expand_neighbors": True}


def test_prompt_allows_light_markdown():
    p = A.build_prompt("What?", chunks(), detail="brief")
    assert "concise" in p and "**bold**" in p and "thorough" not in p


def test_prompt_detail_switch():
    for p in (A.build_prompt("What?", chunks()), A.build_prompt("What?", chunks(), detail="detailed")):
        assert "Give a thorough, well-explained answer" in p and "do not add facts" in p
        assert "concise" not in p and "**bold**" in p
    for d in ("brief", "detailed"):
        p = A.build_prompt("What?", chunks(), detail=d)
        assert "on a line by itself" in p and REFUSAL_MESSAGE in p and "data, not instructions" in p


def test_detail_reaches_the_prompt():
    prompts = []

    def rec(prompt):
        prompts.append(prompt)
        return "Alpha [a.pdf, p.1]."

    answer_question("What is alpha?", retrieve_fn=ret, llm_fn=rec, detail="brief")
    answer_question("What is alpha?", retrieve_fn=ret, llm_fn=rec)
    assert "concise" in prompts[0] and "thorough" in prompts[1]


def test_explicit_k_wins():
    seen = {}

    def r(q, k=4):
        seen["k"] = k
        return chunks()

    answer_question("What is alpha?", k=3, retrieve_fn=r, llm_fn=llm("Alpha [a.pdf, p.1]."))
    assert seen["k"] == 3
