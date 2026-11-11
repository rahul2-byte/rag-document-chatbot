"""Tests for the JSONL question/feedback log."""
import threading

import pytest

from src.qlog import QuestionLog


def test_log_read_stats_and_feedback(tmp_path):
    log = QuestionLog(tmp_path / "logs")
    assert log.read() == {"stats": {"total": 0, "answered": 0, "refused": 0, "smalltalk": 0, "errors": 0,
                                    "up": 0, "down": 0}, "items": []}
    a = log.log_question("q1", "answer", 120, citations=["a.pdf p.1"])
    b = log.log_question("q2", "refusal", 80, rewritten="q2 standalone")
    log.log_question("hi", "smalltalk", 1, intent="greeting")
    log.log_question("x" * 3000, "error", 0)
    assert log.add_feedback(a, "down") and log.add_feedback(a, "up") and log.add_feedback(b, "down")
    assert not log.add_feedback("nope", "up")
    with pytest.raises(ValueError):
        log.add_feedback(a, "meh")
    out = log.read(limit=50)
    assert out["stats"] == {"total": 4, "answered": 1, "refused": 1, "smalltalk": 1, "errors": 1, "up": 1, "down": 1}
    items = out["items"]
    assert [i["question"][:2] for i in items] == ["xx", "hi", "q2", "q1"]  # newest first
    assert len(items[0]["question"]) == 2000
    assert items[3] == {"id": a, "ts": items[3]["ts"], "question": "q1", "rewritten": None, "kind": "answer",
                        "latency_ms": 120, "citations": ["a.pdf p.1"], "feedback": "up"}
    assert items[2]["rewritten"] == "q2 standalone" and items[2]["feedback"] == "down"
    assert len(log.read(limit=2)["items"]) == 2


def test_reader_skips_bad_lines(tmp_path):
    log = QuestionLog(tmp_path)
    qid = log.log_question("good", "answer", 5)
    with log.questions_path.open("a") as f:
        f.write("{not json\n[1,2]\n\n")
    log.feedback_path.write_text('garbage\n{"id": "%s", "rating": "up"}\n{"id": "x", "rating": "sideways"}\n' % qid)
    out = log.read()
    assert out["stats"]["total"] == 1 and out["items"][0]["feedback"] == "up"


def test_thread_safe_appends(tmp_path):
    log = QuestionLog(tmp_path)
    threads = [threading.Thread(target=lambda: [log.log_question("q", "answer", 1) for _ in range(50)])
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert log.read(limit=1000)["stats"]["total"] == 400
