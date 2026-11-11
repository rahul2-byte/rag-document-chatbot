"""Tests for eval grading."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from run_eval import grade  # noqa: E402
from src.answer import AnswerResult  # noqa: E402

Q = {"id": "q1", "answerable": True, "expected": [{"source": "a.pdf", "pages": [3, 4]}]}
U = {"id": "q9", "answerable": False, "expected": []}


def res(**kw) -> AnswerResult:
    base = dict(answer="x", cited_chunks=[], retrieved=[], valid_citations=[("a.pdf", 3)],
                invalid_citations=[], refused=False, error=None)
    base.update(kw)
    return AnswerResult(**base)


def test_pass():
    assert grade(Q, res())[0]


def test_wrong_page():
    assert not grade(Q, res(valid_citations=[("a.pdf", 9)]))[0]


def test_wrong_source():
    assert not grade(Q, res(valid_citations=[("b.pdf", 3)]))[0]


def test_invalid_citation():
    assert not grade(Q, res(invalid_citations=[("a.pdf", 99)]))[0]


def test_no_citation():
    assert not grade(Q, res(valid_citations=[]))[0]


def test_refused_answerable():
    assert not grade(Q, res(refused=True))[0]


def test_unanswerable_refused():
    assert grade(U, res(refused=True, valid_citations=[]))[0]


def test_answered_unanswerable():
    assert not grade(U, res())[0]


def test_error():
    ok, reason = grade(U, res(refused=False, error="no key"))
    assert not ok and "no key" in reason
    assert not grade(Q, res(error="boom"))[0]
