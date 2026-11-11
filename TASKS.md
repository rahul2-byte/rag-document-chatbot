# RAG demo – task tracker

Run everything from repo root. Package code lives in `src/` (a package; run as `python -m src.ingest`, `python -m src.app`).
Venv: `/home/zeek/Project/chatbot/.venv` (created by the user with `uv sync`). Agents never install packages.

## Tasks

| # | Task | Agent | Files owned | Status | Output |
|---|------|-------|-------------|--------|--------|
| 0 | Scaffold, pyproject, uv.lock, requirements.txt, contracts | main | pyproject.toml, uv.lock, requirements.txt, TASKS.md, .gitignore | done | – |
| 1 | Download 3 public-domain docs, data/README.md, eval/questions.json | A (haiku) | data/*, eval/questions.json | todo | data/, eval/questions.json |
| 2 | Ingest + retrieve + their tests | B (sonnet) | src/ingest.py, src/retrieve.py, tests/test_ingest.py, tests/test_retrieve.py | todo | – |
| 3 | Answer + LLM wrapper + citation check + tests | C (sonnet) | src/answer.py, tests/test_answer.py | todo | – |
| 4 | Gradio app + eval runner | D (sonnet) | src/app.py, eval/run_eval.py | todo | – |
| 5 | Integrate, ingest real data, run tests | main | – | todo | – |
| 6 | Run eval with real LLM (needs user API key) | user + main | – | todo | eval output |
| 7 | README.md with real numbers | sonnet | README.md | todo | – |
| 8 | Clean-clone end-to-end check | user + main | – | todo | – |

## Shared contracts (do not change signatures without telling main)

### src/ingest.py
```python
DATA_DIR = Path("data"); INDEX_DIR = Path("index")
CHUNK_SIZE = 800; CHUNK_OVERLAP = 100
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

@dataclass
class Chunk:
    chunk_id: int
    source: str   # file name only, e.g. "NIST.AI.100-1.pdf"
    page: int     # 1-based PDF page; 1 for .txt/.md
    text: str

def load_documents(data_dir: Path) -> list[tuple[str, int, str]]   # (source, page, text) per page; .pdf/.txt/.md; skips README.md
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]
def chunk_documents(pages: list[tuple[str, int, str]]) -> list[Chunk]   # chunks never cross pages
def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray  # float32, L2-normalised, shape (n, d); model loaded once (cached)
def build_index(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR, embed_fn=embed_texts) -> int  # returns #chunks
# writes index_dir/faiss.index (IndexFlatIP) and index_dir/chunks.json (list of Chunk dicts, same order as vectors)
# `python -m src.ingest` rebuilds from scratch (deletes old files first)
```

### src/retrieve.py
```python
class IndexNotFoundError(RuntimeError)   # index missing or has 0 vectors; message tells user to run `python -m src.ingest`

@dataclass
class RetrievedChunk:
    chunk_id: int; source: str; page: int; text: str; score: float

TOP_K = 4
def load_index(index_dir: Path = INDEX_DIR) -> tuple[faiss.Index, list[Chunk]]   # cached per path; raises IndexNotFoundError
def retrieve(question: str, k: int = TOP_K, index_dir: Path = INDEX_DIR, embed_fn=embed_texts) -> list[RetrievedChunk]
# embed_fn called as embed_fn([question], is_query=True)
```

### src/answer.py
```python
REFUSAL_MESSAGE = "I cannot find this in the provided documents."
class LLMError(RuntimeError)   # missing key / missing model / unknown provider / API failure – user-readable message

def generate(prompt: str) -> str
# env: LLM_PROVIDER = anthropic | openai (default anthropic); LLM_MODEL (required);
#      ANTHROPIC_API_KEY or OPENAI_API_KEY. Raises LLMError with clear message.

def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str
def extract_citations(text: str) -> list[tuple[str, int]]          # parses "[file, p.N]"
def validate_citations(text: str, chunks: list[RetrievedChunk]) -> tuple[list[tuple[str,int]], list[tuple[str,int]]]  # (valid, invalid), deduped, order kept
def is_refusal(text: str) -> bool

@dataclass
class AnswerResult:
    answer: str                         # final text shown to user (invalid citations removed + warning note)
    cited_chunks: list[RetrievedChunk]  # retrieved chunks whose (source, page) was validly cited = "passages used"
    retrieved: list[RetrievedChunk]
    valid_citations: list[tuple[str,int]]
    invalid_citations: list[tuple[str,int]]
    refused: bool
    error: str | None = None            # set when something failed; answer then holds a user-readable message

def answer_question(question: str, k: int = TOP_K, retrieve_fn=retrieve, llm_fn=generate) -> AnswerResult
# Never raises for: empty question, missing index, LLM errors -> error set, refused False.
# Non-refusal answer with zero valid citations -> rejected: answer = REFUSAL_MESSAGE, refused True.
```

### eval/questions.json
```json
[{"id": "q1", "question": "...", "answerable": true,
  "expected": [{"source": "file.pdf", "pages": [12, 13]}]},
 {"id": "q9", "question": "...", "answerable": false, "expected": []}]
```
