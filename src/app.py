"""Chat web app for the document Q&A (RAG) demo. Run: python -m src.app"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware
from urllib.parse import urlparse

from src import intents
from src.answer import (DEFAULT_DETAIL, REFUSAL_MESSAGE, AnswerResult, answer_question, answer_question_stream,
                        clean_history, normalize_detail, suggest_followups)
from src.ingest import DATA_DIR, INDEX_DIR, ROOT, _data_files, index_info
from src.retrieve import merge_overlap
from src.qlog import RATINGS, QuestionLog

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
CRASH_MESSAGE = "Something went wrong while answering. Please try again."
LOG_DIR = ROOT / "logs"
MAX_QUESTION_CHARS = 2000
EMPTY_MESSAGE = "Please enter a question."
TOO_LONG_MESSAGE = f"Your question is too long (maximum {MAX_QUESTION_CHARS} characters)."
MAX_FOLLOWUP_INPUT_CHARS = 2000
DOC_MEDIA_TYPES = {".pdf": "application/pdf", ".txt": "text/plain; charset=utf-8",
                   ".md": "text/plain; charset=utf-8"}


DEFAULT_HOSTS = ["127.0.0.1", "localhost"]


def harden(app: FastAPI, allowed_hosts: list[str] | None = None) -> None:
    """Localhost-only protections: Host allow-list (400) and same-origin check for POST/DELETE (403)."""

    @app.middleware("http")
    async def same_origin(request, call_next):
        origin = request.headers.get("origin")
        if request.method in ("POST", "DELETE", "PUT", "PATCH") and origin is not None:
            if urlparse(origin).netloc != request.headers.get("host", ""):
                return JSONResponse({"detail": "Cross-origin request rejected."}, status_code=403)
        return await call_next(request)

    # added last so it runs first: a foreign Host is refused before anything else
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or DEFAULT_HOSTS)


class AskRequest(BaseModel):
    question: str = ""
    history: list | None = None  # [{"role": "user"|"assistant", "content": str}]; cleaned leniently
    detail: str | None = DEFAULT_DETAIL  # "detailed" | "brief"; anything else -> "detailed"


class FollowupsRequest(BaseModel):
    question: str = ""
    answer: str = ""


class FeedbackRequest(BaseModel):
    id: str = ""
    rating: str = ""


def result_kind(result: AnswerResult) -> str:
    if result.error:
        return "error"
    return "refusal" if result.refused else "answer"


def _join_texts(a: str, b: str) -> str:
    """Join two passages of one page, dropping any overlap (neighbour expansion may already include b's start)."""
    a, b = a.strip(), b.strip()
    if not a or b in a:
        return a or b
    return merge_overlap(a, b, max_overlap=len(b))


def merge_sources(chunks) -> list[dict]:
    """One source per (source, page): texts joined in chunk order without repeated text; best score; first-cited order."""
    groups: dict[tuple[str, int], list] = {}
    for c in chunks or []:
        groups.setdefault((c.source, c.page), []).append(c)
    out = []
    for i, ((source, page), cs) in enumerate(groups.items(), 1):
        text = ""
        for c in sorted(cs, key=lambda c: c.chunk_id):
            text = _join_texts(text, c.text)
        out.append({"n": i, "source": source, "page": page,
                    "score": round(max(float(c.score) for c in cs), 4), "text": text})
    return out


def serialize_result(result: AnswerResult) -> dict:
    """JSON-ready dict for the chat client; sources are the cited passages, one per (source, page)."""
    sources = merge_sources(result.cited_chunks)
    kind = result_kind(result)
    answer = result.answer or ""
    if kind == "refusal":
        answer = REFUSAL_MESSAGE
    return {"kind": kind, "answer": answer, "refused": bool(result.refused),
            "error": result.error, "sources": sources,
            "suggestions": list(intents.SUGGESTIONS) if kind == "refusal" else [],
            "rewritten": result.rewritten or None}


def _plain(kind: str, answer: str, *, error: str | None = None, suggestions: list[str] | None = None) -> dict:
    return {"kind": kind, "answer": answer, "refused": False, "error": error, "sources": [],
            "suggestions": suggestions or [], "rewritten": None}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(answer_fn: Callable[..., AnswerResult] = answer_question,
               index_dir: Path = INDEX_DIR, data_dir: Path = DATA_DIR,
               allowed_hosts: list[str] | None = None, log_dir: Path = LOG_DIR,
               stream_fn: Callable = answer_question_stream,
               followups_fn: Callable[..., list[str]] = suggest_followups) -> FastAPI:
    """Build the chat FastAPI app (not started).

    answer_fn(question, history=[...], detail=...) -> AnswerResult;
    stream_fn(question, history=[...], detail=...) -> iterator of meta/delta/done events (see answer_question_stream);
    followups_fn(question, answer, titles) -> [str]."""
    app = FastAPI(title="Document Q&A", docs_url=None, redoc_url=None, openapi_url=None)
    harden(app, allowed_hosts)
    qlog = QuestionLog(log_dir)

    @app.get("/api/info")
    def info() -> dict:
        try:
            i = index_info(index_dir, data_dir)
        except Exception:  # never block the page on metadata
            logger.exception("index_info failed")
            i = {"exists": False, "num_documents": 0, "num_chunks": 0, "documents": {},
                 "built_at": None, "stale": True}
        return {"exists": bool(i["exists"]), "num_documents": i["num_documents"],
                "num_chunks": i["num_chunks"], "documents": sorted(i["documents"]),
                "built_at": i["built_at"], "stale": bool(i["stale"]),
                "examples": list(intents.EXAMPLE_QUESTIONS), "max_question_chars": MAX_QUESTION_CHARS}

    def early_reply(q: str) -> dict | None:
        """Smalltalk / input-error reply (no LLM), or None when the question needs answering."""
        reply = intents.handle(q, index_dir, data_dir) if len(q) <= MAX_QUESTION_CHARS else None
        if reply is not None:
            return {**_plain("smalltalk", reply.answer, suggestions=reply.suggestions), "_intent": reply.intent}
        if not q:
            return _plain("error", EMPTY_MESSAGE, error=EMPTY_MESSAGE, suggestions=list(intents.SUGGESTIONS))
        if len(q) > MAX_QUESTION_CHARS:
            return _plain("error", TOO_LONG_MESSAGE, error=TOO_LONG_MESSAGE)
        return None

    def from_result(res: AnswerResult) -> dict:
        out = serialize_result(res)
        out["_cites"] = [f"{s} p.{p}" for s, p in (res.valid_citations or [])] if out["kind"] == "answer" else []
        return out

    def log_done(question: str, result: dict, start: float, qid: str | None = None) -> dict:
        """Pop private fields, log once, set result["id"]."""
        intent = result.pop("_intent", None)
        cites = result.pop("_cites", [])
        latency = int((time.monotonic() - start) * 1000)
        result["id"] = qlog.log_question(question, result["kind"], latency, rewritten=result.get("rewritten"),
                                         citations=cites, intent=intent, qid=qid)
        return result

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        start = time.monotonic()
        question = req.question or ""
        try:
            history = clean_history(req.history)
            result = respond(question, history, normalize_detail(req.detail))
        except Exception:  # last resort: never crash the UI
            logger.exception("answer_fn failed")
            result = _plain("error", CRASH_MESSAGE, error=CRASH_MESSAGE)
        return log_done(question, result, start)

    def respond(question: str, history: list[dict], detail: str) -> dict:
        q = question.strip()
        early = early_reply(q)
        if early is not None:
            return early
        return from_result(answer_fn(q, history=history, detail=detail))

    @app.post("/api/ask/stream")
    def ask_stream(req: AskRequest) -> StreamingResponse:
        start = time.monotonic()
        question = req.question or ""
        qid = uuid.uuid4().hex

        def events():
            q = question.strip()
            try:
                result = early_reply(q)
                if result is None:
                    result = _plain("error", CRASH_MESSAGE, error=CRASH_MESSAGE)
                    gen = stream_fn(q, history=clean_history(req.history), detail=normalize_detail(req.detail))
                    for ev in gen:
                        kind = ev.get("type")
                        if kind == "meta":
                            yield _sse({"type": "meta", "id": qid, "rewritten": ev.get("rewritten") or None})
                        elif kind == "delta":
                            yield _sse({"type": "delta", "text": ev.get("text", "")})
                        elif kind == "done":
                            result = from_result(ev["result"])
                            break
            except Exception:  # last resort: the stream always ends with a done event
                logger.exception("stream_fn failed")
                result = _plain("error", CRASH_MESSAGE, error=CRASH_MESSAGE)
            done = log_done(question, result, start, qid)
            yield _sse({"type": "done", **done})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/followups")
    def followups(req: FollowupsRequest) -> dict:
        q, a = (req.question or "").strip(), (req.answer or "").strip()
        if not q or not a or len(q) > MAX_FOLLOWUP_INPUT_CHARS:
            return {"followups": []}
        try:
            titles = [t for _, t in intents.document_titles(index_dir, data_dir)]
            out = followups_fn(q, a[:MAX_FOLLOWUP_INPUT_CHARS], titles)
            return {"followups": [x for x in (out or []) if isinstance(x, str)][:3]}
        except Exception:  # suggestions are optional
            logger.exception("followups_fn failed")
            return {"followups": []}

    @app.get("/docs/{name}")
    def document(name: str) -> FileResponse:
        path = next((p for p in _data_files(data_dir) if p.name == name), None)
        if path is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        return FileResponse(path, media_type=DOC_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                            filename=path.name, content_disposition_type="inline")

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        if req.rating not in RATINGS:
            raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'.")
        if not qlog.add_feedback(req.id, req.rating):
            raise HTTPException(status_code=404, detail="Unknown question id.")
        return {"ok": True}

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(WEB_DIR / "chat" / "index.html")

    app.mount("/shared", StaticFiles(directory=WEB_DIR / "shared"), name="shared")
    app.mount("/static", StaticFiles(directory=WEB_DIR / "chat"), name="static")
    return app


def warm_up() -> None:
    """Load the index and embed one dummy query so the first real question is fast."""
    try:
        from src.ingest import embed_texts
        from src.retrieve import load_index

        load_index()
        embed_texts(["warm up"], is_query=True)
    except Exception as exc:  # missing index etc.: keep going
        logger.warning("Warm-up skipped: %s", exc)


def main() -> None:
    """Serve on 127.0.0.1 (port from $PORT, default 7860)."""
    logging.basicConfig(level=logging.INFO)
    if not (INDEX_DIR / "faiss.index").exists():
        logger.warning("%s/faiss.index is missing; run `python -m src.ingest` first.", INDEX_DIR)
    warm_up()
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()
