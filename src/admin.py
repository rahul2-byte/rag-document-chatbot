"""Owner console (FastAPI): manage the document library, rebuild the index, preview retrieval."""
from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ingest import (
    DATA_DIR,
    INDEX_DIR,
    SUPPORTED_EXTS,
    build_index,
    embed_texts,
    index_info,
    list_documents,
)
from src.app import LOG_DIR, harden
from src.qlog import QuestionLog
from src.retrieve import TOP_K, IndexNotFoundError, retrieve

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_PORT = 7861
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
PREVIEW_CHARS = 300

_rebuild_lock = threading.Lock()


def _human_size(n: int) -> str:
    """Format a byte count as KB or MB."""
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{max(n, 1) / 1024:.1f} KB" if n else "0.0 KB"


def _safe_name(raw: str) -> str:
    """Reduce a client-supplied file name to a safe base name (may be empty)."""
    base = Path(str(raw).replace("\\", "/")).name
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^\w.\- ]+", "_", stem).strip(" .")
    return f"{stem}{ext.lower()}" if stem else ""


def save_uploads(files: Sequence[tuple[str, bytes]] | None, data_dir: Path = DATA_DIR) -> dict:
    """Save (filename, content) pairs into data_dir; return {"saved": [...], "errors": [...]}."""
    data_dir = Path(data_dir)
    saved: list[dict] = []
    errors: list[dict] = []
    if not files:
        return {"saved": saved, "errors": [{"name": "", "message": "No files selected."}]}
    data_dir.mkdir(parents=True, exist_ok=True)

    def reject(shown: str, msg: str) -> None:
        errors.append({"name": shown, "message": msg})

    for raw_name, content in files:
        shown = Path(str(raw_name).replace("\\", "/")).name or str(raw_name)
        try:
            name = _safe_name(raw_name)
            if not name or Path(name).suffix not in SUPPORTED_EXTS:
                reject(shown, "Unsupported type. Use .pdf, .txt or .md.")
                continue
            if name.lower() == "readme.md":
                reject(shown, "README.md is reserved and not indexed.")
                continue
            if len(content) == 0:
                reject(shown, "File is empty.")
                continue
            if len(content) > MAX_UPLOAD_BYTES:
                reject(shown, "Larger than 50 MB.")
                continue
            dest = data_dir / name
            existed = dest.exists()
            tmp = data_dir / f".upload-{os.getpid()}-{name}.tmp"
            tmp.write_bytes(content)
            if name.endswith(".pdf"):
                from pypdf import PdfReader

                try:
                    if len(PdfReader(str(tmp)).pages) == 0:
                        raise ValueError("no pages")
                except Exception:  # noqa: BLE001 - any parse failure means unusable
                    tmp.unlink(missing_ok=True)
                    reject(shown, "Cannot be opened as a PDF.")
                    continue
            os.replace(tmp, dest)
            saved.append({"name": name, "replaced": existed})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Upload failed for %s", shown)
            reject(shown, f"Failed: {exc}")
    return {"saved": saved, "errors": errors}


def delete_document(name: str | None, data_dir: Path = DATA_DIR) -> dict:
    """Delete a document, accepting only names returned by list_documents()."""
    known = {d["name"] for d in list_documents(data_dir)}
    if not name or name not in known:
        return {"ok": False, "message": "Not a document in the library."}
    try:
        (Path(data_dir) / name).unlink()
    except OSError as exc:
        logger.exception("Delete failed")
        return {"ok": False, "message": f"Could not delete {name}: {exc}"}
    return {"ok": True, "message": f"Deleted {name}."}


def rebuild(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR, embed_fn: Callable = embed_texts) -> dict:
    """Rebuild the index; returns {"ok", "num_chunks", "seconds", "message", "busy"}."""
    if not _rebuild_lock.acquire(blocking=False):
        return {"ok": False, "busy": True, "num_chunks": 0, "seconds": 0.0,
                "message": "A rebuild is already running. Please wait."}
    try:
        start = time.monotonic()
        try:
            n_chunks = build_index(data_dir, index_dir, embed_fn)
        except Exception as exc:  # noqa: BLE001 - never show a traceback
            logger.exception("Rebuild failed")
            return {"ok": False, "busy": False, "num_chunks": 0,
                    "seconds": round(time.monotonic() - start, 1), "message": f"Rebuild failed: {exc}"}
        seconds = round(time.monotonic() - start, 1)
        n_docs = index_info(index_dir, data_dir)["num_documents"]
        return {"ok": True, "busy": False, "num_chunks": n_chunks, "seconds": seconds,
                "message": f"Indexed {n_chunks} passages from {n_docs} documents in {seconds:.1f}s."}
    finally:
        _rebuild_lock.release()


def status(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR) -> dict:
    """Index info plus the document list (with per-file index membership)."""
    info = index_info(index_dir, data_dir)
    docs = []
    for d in list_documents(data_dir):
        entry = info["documents"].get(d["name"])
        docs.append({"name": d["name"], "type": d["type"], "pages": d["pages"],
                     "size": d["size_bytes"], "chunks": entry["chunks"] if entry else 0,
                     "modified": d["modified"], "in_index": bool(entry)})
    return {"exists": info["exists"], "built_at": info["built_at"], "num_documents": info["num_documents"],
            "num_chunks": info["num_chunks"], "embed_model": info["embed_model"], "stale": info["stale"],
            "documents": docs}


def retrieval_preview(question: str | None, index_dir: Path = INDEX_DIR,
                      embed_fn: Callable = embed_texts) -> dict:
    """Top passages for a question (no LLM call): {"results": [...], "message": str|None}."""
    if not question or not question.strip():
        return {"results": [], "message": "Type a question first."}
    try:
        chunks = retrieve(question, k=TOP_K, index_dir=index_dir, embed_fn=embed_fn)
    except IndexNotFoundError:
        return {"results": [], "message": "No index yet: add documents and click Rebuild."}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Preview failed")
        return {"results": [], "message": f"Preview failed: {exc}"}
    results = [{"n": i, "source": c.source, "page": c.page, "score": round(float(c.score), 4),
                "text": c.text.strip()} for i, c in enumerate(chunks, 1)]
    return {"results": results, "message": None if results else "No passages found."}


class PreviewRequest(BaseModel):
    question: str = ""


def create_admin_app(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR,
                     embed_fn: Callable = embed_texts,
                     allowed_hosts: list[str] | None = None, log_dir: Path = LOG_DIR) -> FastAPI:
    """Build the owner console FastAPI app (not started)."""
    app = FastAPI(title="Owner console", docs_url=None, redoc_url=None, openapi_url=None)
    harden(app, allowed_hosts)
    qlog = QuestionLog(log_dir)

    def reject_if_rebuilding() -> None:
        if _rebuild_lock.locked():
            raise HTTPException(status_code=409, detail="A rebuild is running. Try again when it finishes.")

    @app.get("/api/status")
    def get_status() -> dict:
        return status(data_dir, index_dir)

    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(default=[])) -> dict:
        reject_if_rebuilding()
        pairs: list[tuple[str, bytes]] = []
        for f in files:
            # read at most limit+1 bytes so oversize files are rejected without loading everything
            pairs.append((f.filename or "", await f.read(MAX_UPLOAD_BYTES + 1)))
        return save_uploads(pairs, data_dir)

    @app.delete("/api/documents/{name}")
    def delete(name: str) -> dict:
        reject_if_rebuilding()
        res = delete_document(name, data_dir)
        if not res["ok"]:
            raise HTTPException(status_code=404, detail=res["message"])
        return res

    @app.post("/api/rebuild")
    def do_rebuild():
        res = rebuild(data_dir, index_dir, embed_fn)
        if res.get("busy"):
            raise HTTPException(status_code=409, detail=res["message"])
        return {k: res[k] for k in ("ok", "num_chunks", "seconds", "message")}

    @app.post("/api/preview")
    def preview(req: PreviewRequest) -> dict:
        return retrieval_preview(req.question, index_dir, embed_fn)

    @app.get("/api/questions")
    def questions(limit: int = Query(50, ge=1, le=1000)) -> dict:
        return qlog.read(limit)

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(WEB_DIR / "admin" / "index.html")

    app.mount("/shared", StaticFiles(directory=WEB_DIR / "shared"), name="shared")
    app.mount("/static", StaticFiles(directory=WEB_DIR / "admin"), name="static")
    return app


def main() -> None:
    """Serve the console on 127.0.0.1 only."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    port = int(os.environ.get("ADMIN_PORT", DEFAULT_PORT))
    uvicorn.run(create_admin_app(), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
