# Document Q&A chatbot (RAG demo)

A small retrieval-augmented chatbot that answers questions only from your documents. Every answer cites its sources as `[file, p.N]`, and the bot refuses when the answer is not in the documents.

## Screenshots

![Chat answer with numbered citations and source chips](docs/screenshots/chat-cited-answer.png)

![Owner console: documents, index status and recent questions](docs/screenshots/owner-console.png)

<p>
  <img src="docs/screenshots/source-pdf-page.png" alt="Source chip opening the original PDF at the cited page" width="49%">
  <img src="docs/screenshots/refusal.png" alt="Refusal for a question not covered by the documents" width="49%">
</p>

A demo video can be generated locally with `node demo/record_demo.cjs` (see `demo/DEMO_SCRIPT.md`); it writes `demo/out/demo.mp4`, which is not committed.

## Features

- Two interfaces: user chat on port 7860 and owner console on port 7861
- Chat: ChatGPT-style centred conversation with numbered inline citation markers and a Sources row of chips under each answer; clicking a chip expands the exact passage (file, page, relevance). Refusals show as a quiet "I couldn't find this in the documents." line. Light and dark mode, mobile layout
- Answers stream token by token over server-sent events (`POST /api/ask/stream`); the Stop button cancels an answer in progress
- Detailed / Brief toggle under the composer (default Detailed): Detailed uses 8 retrieved passages and asks for a thorough explanation; Brief uses 4 and asks for a concise answer
- Source chips expand to the exact passage and have an "Open page N" link that opens the original PDF at that page (`GET /docs/<file>#page=N`; only files in `data/` are served). Sources from the same page are merged into one entry
- Follow-up questions are understood in context: the question is rewritten into a standalone one using recent chat history, and the UI shows "Searched for: ..."
- Up to 3 "Related" follow-up questions are suggested after each answer (`POST /api/followups`); click one to ask it
- Comparison questions (e.g. "compare X and Y") retrieve passages for each side
- Built-in replies with no LLM call for greetings, thanks, help, "list documents", too-short input and prompt-injection attempts. Off-topic questions are refused with the same "I couldn't find this in the documents." line plus example questions
- Thumbs up/down feedback, copy and regenerate on each answer; the conversation is kept in the browser's localStorage
- Owner console: upload PDF/TXT/MD (50 MB limit per file), document list, delete with confirm, rebuild index, index status with an out-of-date badge, retrieval preview. Upload and delete are blocked while a rebuild runs
- Owner console "Recent questions" card: questions asked in the chat (`logs/questions.jsonl`, localhost only) with an answered % over document questions, plus user feedback (`logs/feedback.jsonl`)
- Local embeddings (`BAAI/bge-small-en-v1.5`) and a FAISS index
- Hybrid retrieval: BM25 keyword search (in-repo, no extra package) fused with FAISS vector search using Reciprocal Rank Fusion (k=60)
- Chunking at 800 characters with 100 overlap, keeping file name and page; chunks never cross pages
- Top-4 passages after fusion for Brief answers, top-8 for Detailed
- Citations validated in code against the retrieved chunks: invalid ones are removed, and answers with no valid citation are rejected
- Refusal when the answer is not in the documents
- Clean error messages for a missing key, an empty index, an empty question, and API failure
- Swappable LLM behind one `generate()` function (openrouter, anthropic, openai)
- Unit tests (209, run with `uv run pytest -q`) and a 10-question eval

## Quick start

```bash
git clone <repo-url> rag-demo && cd rag-demo
uv sync
cp .env.example .env    # then put your OPENROUTER_API_KEY in .env
uv run python -m src.ingest
uv run --env-file .env python -m src.app      # open http://127.0.0.1:7860
```

The owner console runs with `uv run --env-file .env python -m src.admin` (http://127.0.0.1:7861). It needs no key to manage documents.

Pip alternative:

```bash
python -m venv .venv && pip install -r requirements.txt
```

## Configuration

Set these in `.env` (loaded with `--env-file .env`). Defaults come from the code.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `openai` or `openrouter`. `.env.example` sets `openrouter`. |
| `LLM_MODEL` | none (required) | Model id for the chosen provider, e.g. `anthropic/claude-haiku-5.5` on OpenRouter |
| `LLM_REASONING` | `off` | `off` or `on`. Applies only when `LLM_PROVIDER=openrouter`. `off` disables the model's hidden reasoning step. Measured on this project, it cut time-to-first-token from about 3.4–3.9 s to about 1.2–1.3 s, with the eval still 10/10 |
| `OPENROUTER_API_KEY` | none | Required when `LLM_PROVIDER=openrouter` |
| `ANTHROPIC_API_KEY` | none | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | none | Required when `LLM_PROVIDER=openai` |
| `PORT` | `7860` | Port for the user chat app (`src.app`) |
| `ADMIN_PORT` | `7861` | Port for the owner console (`src.admin`) |

Both apps bind to `127.0.0.1` and accept only `Host` 127.0.0.1 or localhost. Cross-origin POST and DELETE requests are rejected with 403.

## How it works

Indexing (`python -m src.ingest`, or the owner console Rebuild button):

```
load -> chunk -> embed -> FAISS
```

Answering (each question):

```
question (in chat, a follow-up is first rewritten into a standalone question) -> embed + BM25 -> FAISS + keyword ranks -> RRF fuse -> top-4 (Brief) or top-8 (Detailed) -> prompt -> LLM -> citation check -> answer + sources
```

Layout:

```
src/
  ingest.py     load, chunk, embed, build the index, index metadata
  retrieve.py   load the FAISS index, BM25 keyword search, RRF fusion, top-4 chunks
  answer.py     prompt, generate() LLM wrapper, citation check, refusal
  intents.py    built-in replies (greetings, help, list documents, injection guard) without an LLM call
  qlog.py       question and feedback log (logs/*.jsonl) for the owner console
  app.py        user chat app (FastAPI, port 7860)
  admin.py      owner console (FastAPI, port 7861)
web/          hand-written HTML/CSS/JavaScript frontend, no build step
  shared/     assets common to both apps
  chat/       user chat page
  admin/      owner console page
tests/          unit tests (pytest)
eval/
  questions.json  10 evaluation questions (8 answerable, 2 unanswerable)
  run_eval.py     runs the questions through the pipeline and grades them
data/           source documents (PDF/TXT/MD) and data/README.md
index/          generated index (faiss.index, chunks.json, meta.json); gitignored
```

## Adding your own documents

Either:

- Use the owner console (http://127.0.0.1:7861): upload files, then click Rebuild index.
- Or drop `.pdf`, `.txt` or `.md` files into `data/` and run `uv run python -m src.ingest`. This rebuilds the index from scratch in one command.

The running chat app picks up a rebuilt index automatically, without a restart.

## Tests and evaluation

```bash
uv run pytest -q
uv run --env-file .env python eval/run_eval.py
```

The eval needs a working LLM key because it calls the real pipeline.

### Latest eval result

Run on 2026-11-11 with LLM_MODEL=anthropic/claude-haiku-5.5 via OpenRouter, LLM_REASONING=off:

```text
q1    ANSWER  PASS  [NIST.AI.100-1.pdf, p.25], [NIST.AI.100-1.pdf, p.8], [NIST.AI.100-1.pdf, p.37] | cites expected page
q2    ANSWER  PASS  [NIST.AI.100-1.pdf, p.17] | cites expected page
q3    ANSWER  PASS  [NIST.CSWP.29.pdf, p.8], [NIST.CSWP.29.pdf, p.10], [NIST.CSWP.29.pdf, p.20] | cites expected page
q4    ANSWER  PASS  [NIST.CSWP.29.pdf, p.31], [NIST.CSWP.29.pdf, p.29] | cites expected page
q5    ANSWER  PASS  [NIST.CSWP.29.pdf, p.27], [NIST.CSWP.29.pdf, p.20], [NIST.CSWP.29.pdf, p.9] | cites expected page
q6    ANSWER  PASS  [NIST.CSWP.01162020.pdf, p.11], [NIST.CSWP.01162020.pdf, p.22] | cites expected page
q7    ANSWER  PASS  [NIST.CSWP.01162020.pdf, p.2], [NIST.CSWP.01162020.pdf, p.5] | cites expected page
q8    ANSWER  PASS  [NIST.AI.100-1.pdf, p.7], [NIST.AI.100-1.pdf, p.26] | cites expected page
q9    REFUSE  PASS  - | correctly refused
q10   REFUSE  PASS  - | correctly refused

Answerable with correct citations: 8/8
Unanswerable refused: 2/2
TOTAL: 10/10
```

LLM output varies between runs, so results can differ from the block above.

## Data and licenses

The sample documents are listed in [data/README.md](data/README.md), with source URLs and license basis:

- `NIST.AI.100-1.pdf`: Artificial Intelligence Risk Management Framework (AI RMF 1.0)
- `NIST.CSWP.29.pdf`: The NIST Cybersecurity Framework (CSF) 2.0
- `NIST.CSWP.01162020.pdf`: NIST Privacy Framework, Version 1.0

These are US government works and are public domain in the United States (17 U.S.C. §105).

The code in this repository is released under the [MIT License](LICENSE). The MIT license does not apply to the PDFs in `data/`, which keep their own public-domain status.

## Changes from the original spec

- The UI is not Gradio. Both apps are FastAPI + uvicorn with a hand-written HTML/CSS/JavaScript frontend in `web/` (no build step, no npm, no CDN; works offline). The user asked for this, overriding the original Gradio requirement.
- Retrieval is hybrid: BM25 keyword search fused with FAISS vector search using Reciprocal Rank Fusion (k=60). The original spec named FAISS only.

## Limitations

- No authentication. Both apps bind to 127.0.0.1 and must not be exposed publicly without adding auth.
- The citation check is at page level: a citation is valid if its file and page match a retrieved passage.
- Conversation memory is limited. A follow-up question is rewritten into a standalone question using recent chat history, and that rewritten question is what retrieval searches.
- Scanned PDFs without a text layer are not supported (no OCR).
