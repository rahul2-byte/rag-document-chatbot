"""Append-only JSONL question log and feedback (logs/questions.jsonl, logs/feedback.jsonl)."""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.ingest import ROOT

logger = logging.getLogger(__name__)

LOG_DIR = ROOT / "logs"
QUESTIONS_FILE = "questions.jsonl"
FEEDBACK_FILE = "feedback.jsonl"
RATINGS = ("up", "down")
KINDS = ("answer", "refusal", "smalltalk", "error")
MAX_LOGGED_CHARS = 2000

_lock = threading.Lock()  # one lock for all logs in this process: appends are tiny


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _append(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def _read(path: Path) -> list[dict]:
    """All well-formed JSON object lines; bad lines are skipped."""
    out: list[dict] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except FileNotFoundError:
        pass
    except OSError:
        logger.exception("Could not read %s", path)
    return out


class QuestionLog:
    """Question + feedback log rooted at `log_dir`."""

    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = Path(log_dir)
        self.questions_path = self.log_dir / QUESTIONS_FILE
        self.feedback_path = self.log_dir / FEEDBACK_FILE

    def log_question(self, question: str, kind: str, latency_ms: int, *, rewritten: str | None = None,
                     citations: list[str] | None = None, intent: str | None = None,
                     qid: str | None = None) -> str:
        """Append one question record; returns its id. Never raises."""
        qid = qid or uuid.uuid4().hex
        record = {"id": qid, "ts": _now(), "question": (question or "")[:MAX_LOGGED_CHARS],
                  "rewritten": rewritten, "kind": kind, "intent": intent,
                  "latency_ms": int(latency_ms), "citations": list(citations or [])}
        try:
            _append(self.questions_path, record)
        except OSError:
            logger.exception("Could not write question log")
        return qid

    def known_ids(self) -> set[str]:
        return {str(r.get("id")) for r in _read(self.questions_path) if r.get("id")}

    def add_feedback(self, qid: str, rating: str) -> bool:
        """Record a rating; False if the id is unknown. Raises ValueError for a bad rating."""
        if rating not in RATINGS:
            raise ValueError("rating must be 'up' or 'down'")
        if not qid or qid not in self.known_ids():
            return False
        _append(self.feedback_path, {"id": qid, "ts": _now(), "rating": rating})
        return True

    def feedback(self) -> dict[str, str]:
        """Latest rating per id."""
        out: dict[str, str] = {}
        for r in _read(self.feedback_path):
            if r.get("rating") in RATINGS and r.get("id"):
                out[str(r["id"])] = r["rating"]
        return out

    def read(self, limit: int = 50) -> dict:
        """{"stats": {...}, "items": [...]} with items newest first."""
        records = _read(self.questions_path)
        fb = self.feedback()
        kinds = [r.get("kind") for r in records]
        stats = {
            "total": len(records),
            "answered": kinds.count("answer"),
            "refused": kinds.count("refusal"),
            "smalltalk": kinds.count("smalltalk"),
            "errors": kinds.count("error"),
            "up": sum(1 for r in records if fb.get(str(r.get("id"))) == "up"),
            "down": sum(1 for r in records if fb.get(str(r.get("id"))) == "down"),
        }
        items = []
        for r in reversed(records[-max(0, limit):] if limit > 0 else []):
            cites = r.get("citations")
            items.append({
                "id": str(r.get("id", "")), "ts": r.get("ts"), "question": str(r.get("question", "")),
                "rewritten": r.get("rewritten"), "kind": r.get("kind"),
                "latency_ms": r.get("latency_ms"),
                "citations": [str(c) for c in cites] if isinstance(cites, list) else [],
                "feedback": fb.get(str(r.get("id"))),
            })
        return {"stats": stats, "items": items}
