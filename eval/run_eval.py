"""Run the evaluation questions through the pipeline. Usage: python eval/run_eval.py"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.answer import AnswerResult, answer_question  # noqa: E402

logger = logging.getLogger(__name__)
QUESTIONS_PATH = ROOT / "eval" / "questions.json"


def grade(question: dict, result: AnswerResult) -> tuple[bool, str]:
    """Grade one result against its question; return (passed, reason)."""
    if result.error:
        return False, f"error: {result.error}"
    if not question.get("answerable"):
        if result.refused:
            return True, "correctly refused"
        return False, "should have refused but answered"
    if result.refused:
        return False, "refused an answerable question"
    if not result.valid_citations:
        return False, "no valid citation"
    if result.invalid_citations:
        return False, f"invalid citations: {result.invalid_citations}"
    for src, page in result.valid_citations:
        for exp in question.get("expected", []):
            if exp.get("source") == src and page in exp.get("pages", []):
                return True, "cites expected page"
    return False, f"cited {result.valid_citations}, expected {question.get('expected')}"


def main() -> int:
    """Run all questions, print a table and summary; return the exit code."""
    logging.basicConfig(level=logging.WARNING)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    ans_pass = ans_total = un_pass = un_total = 0
    for q in questions:
        try:
            result = answer_question(q["question"])
            ok, reason = grade(q, result)
            cites = ", ".join(f"[{s}, p.{p}]" for s, p in result.valid_citations) or "-"
        except Exception as exc:  # keep going; report per row
            logger.exception("question %s crashed", q.get("id"))
            ok, reason, cites = False, f"error: {exc}", "-"
        answerable = bool(q.get("answerable"))
        if answerable:
            ans_total += 1
            ans_pass += ok
        else:
            un_total += 1
            un_pass += ok
        print(f"{q['id']:<5} {'ANSWER' if answerable else 'REFUSE':<7} "
              f"{'PASS' if ok else 'FAIL':<5} {cites} | {reason}")
    total = ans_total + un_total
    print()
    print(f"Answerable with correct citations: {ans_pass}/{ans_total}")
    print(f"Unanswerable refused: {un_pass}/{un_total}")
    print(f"TOTAL: {ans_pass + un_pass}/{total}")
    return 0 if ans_pass + un_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
