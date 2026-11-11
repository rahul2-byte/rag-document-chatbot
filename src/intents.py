"""Deterministic (no LLM) detection of small-talk intents and their replies.

Matching is deliberately narrow: most intents must match the *whole* message (after
lower-casing and trimming punctuation), so real document questions such as
"Help me understand the GOVERN function" are never intercepted. Only prompt-injection
phrases are searched anywhere in the message.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.ingest import CHUNKS_FILE, DATA_DIR, INDEX_DIR, index_info

EXAMPLE_QUESTIONS = [
    "What are the four core functions of the AI RMF?",
    "What are the five Functions of the NIST Privacy Framework?",
    "What are the six Functions of the CSF 2.0?",
    "What fines does the NIST Privacy Framework prescribe for GDPR violations?",
]
# Suggestions offered as buttons: the answerable examples only.
SUGGESTIONS = EXAMPLE_QUESTIONS[:3]

INTENTS = ("greeting", "thanks", "goodbye", "help", "list_documents", "too_short", "prompt_injection")


@dataclass
class IntentReply:
    intent: str
    answer: str
    suggestions: list[str] = field(default_factory=lambda: list(SUGGESTIONS))


# ------------------------------------------------------------------ matching
def _norm(text: str) -> str:
    """Lower-case, unify apostrophes, drop surrounding punctuation/emoji, collapse spaces."""
    t = text.lower().replace("’", "'").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    return t


_POLITE = r"(?: (?:there|all|everyone|bot|again|friend|so much|a lot|very much|a bunch|please|pls))*"
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("greeting", re.compile(
        r"^(?:hi+|hello+|hey+|hiya|howdy|greetings|yo|hallo|namaste|sup|what'?s up|good (?:morning|afternoon|evening|day)"
        r"|(?:hi|hello|hey),? how are you(?: doing)?|how are you(?: doing)?)" + _POLITE + r"$")),
    ("thanks", re.compile(
        r"^(?:(?:ok(?:ay)?|great|cool|perfect|awesome|nice)[ ,!]+)?(?:thanks|thank you|thank u|thx|ty|cheers|"
        r"much appreciated|appreciate it|thanks? a lot)" + _POLITE + r"$")),
    ("goodbye", re.compile(
        r"^(?:(?:ok(?:ay)?|thanks|thank you)[ ,!]+)?(?:bye+|bye bye|goodbye|good bye|see you(?: later| soon)?|see ya|cya"
        r"|good night|later|farewell|take care)" + _POLITE + r"$")),
    ("help", re.compile(
        r"^(?:help|help me|help please|please help|i need help|menu|instructions|"
        r"what can you do|what do you do|what are you|who are you|what is this|what's this|"
        r"how do i use (?:this|you|it)|how (?:does|do) (?:this|you|it) work|how to use (?:this|you|it)|"
        r"what can i ask(?: you)?|what should i ask(?: you)?|what do you know|"
        r"what can you (?:help|answer) (?:me )?with|what can you help me with)" + _POLITE + r"$")),
    ("list_documents", re.compile(
        r"^(?:(?:please |can you |could you )?(?:list|show|name)(?: me)?(?: all)?(?: the| your)? "
        r"(?:documents|docs|files|sources|pdfs)(?: you have| available| indexed)?"
        r"|(?:what|which) (?:documents|docs|files|sources|pdfs) (?:do you have|are (?:there|available|indexed|loaded)"
        r"|do you (?:know|cover|use)|can you (?:search|answer from))"
        r"|how many (?:documents|docs|files|pdfs)(?: do you have| are there| are indexed)?"
        r"|(?:documents|docs|files|sources))" + _POLITE + r"$")),
]
_FILLER = {
    "ok", "okay", "k", "kk", "yes", "no", "yep", "nope", "sure", "hmm", "hm", "cool", "nice", "lol", "great",
    "fine", "alright", "test", "testing", "ah", "oh", "uh", "um", "huh", "wow", "right", "yeah", "nah", "what",
}
_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b(?: \w+){0,3} (?:previous|prior|above|earlier|your|all|the|system)"
    r"(?: \w+){0,2} (?:instructions?|prompts?|rules|guidelines|directions)"
    r"|\b(?:reveal|show|print|repeat|output|display|leak|tell me|give me|what (?:is|are)|share)\b(?: \w+){0,3} "
    r"(?:system prompt|system message|initial prompt|hidden prompt|original prompt|(?:your|the) (?:instructions|prompt|rules))"
    r"|\byou are now\b|\bfrom now on,? you (?:are|will)\b|\bjailbreak|\bdan mode\b|\bdeveloper mode\b"
    r"|\bpretend (?:you are|to be)\b|\bdo anything now\b",
)


def _is_gibberish(raw: str, t: str) -> bool:
    """Message without letters, filler words, or a single keyboard-mash token."""
    if not re.search(r"[a-z]", t):
        return True
    if t in _FILLER:
        return True
    if " " in t or not t.isalpha():
        return False
    if len(t) < 2:
        return True
    if raw.strip(" ?!.").isupper():
        return False  # acronym such as "CSF" or "RMF": let retrieval try
    if any(t in row for row in _KEYBOARD_ROWS) and len(t) >= 3:
        return True
    if not re.search(r"[aeiouy]", t) and len(t) >= 3:
        return True
    return bool(re.fullmatch(r"(.)\1{2,}", t))


def detect_intent(message: str | None) -> str | None:
    """Return the intent name for a small-talk message, or None for a real question."""
    if not message or not message.strip():
        return None
    raw = message.strip()
    t = _norm(raw)
    if _INJECTION.search(re.sub(r"\s+", " ", raw.lower().replace("’", "'"))):
        return "prompt_injection"
    if _is_gibberish(raw, t):
        return "too_short"
    for name, pattern in _PATTERNS:
        if pattern.match(t):
            return name
    return None


# --------------------------------------------------------------- doc titles
_TITLE_RE = re.compile(
    r"((?:NIST\s+)?(?:[A-Za-z]+\s+){1,5}Framework(?:\s*\([^()]{1,20}\))?(?:\s+\d+\.\d+)?)", re.IGNORECASE
)
_KEEP_UPPER = {"NIST", "AI", "RMF", "CSF"}


def _tidy_title(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.IGNORECASE)
    if t.isupper():
        t = " ".join(w if w in _KEEP_UPPER else w.capitalize() for w in t.split())
    return t


def title_from_text(text: str) -> str | None:
    """Best-effort readable title ("... Framework (ABBR) 1.0") from a document's first passage."""
    m = _TITLE_RE.search(text or "")
    return _tidy_title(m.group(1)) if m else None


@lru_cache(maxsize=8)
def _titles_cached(path: str, mtime_ns: int) -> dict[str, str]:
    titles: dict[str, str] = {}
    try:
        chunks = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return titles
    for c in chunks:
        src = c.get("source")
        if src and src not in titles:
            titles[src] = title_from_text(c.get("text", "")) or src
    return titles


def document_titles(index_dir: Path = INDEX_DIR, data_dir: Path = DATA_DIR) -> list[tuple[str, str]]:
    """[(file name, readable title)] for every indexed document, sorted by file name."""
    try:
        names = sorted(index_info(index_dir, data_dir)["documents"])
    except Exception:  # noqa: BLE001 - metadata must never break small talk
        names = []
    path = Path(index_dir) / CHUNKS_FILE
    try:
        titles = _titles_cached(str(path.resolve()), path.stat().st_mtime_ns)
    except OSError:
        titles = {}
    return [(n, titles.get(n, n)) for n in names]


# ------------------------------------------------------------------ replies
def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _covers(docs: list[tuple[str, str]]) -> str:
    if not docs:
        return "No documents are indexed yet, so there is nothing for me to search right now."
    return f"I answer questions about {_join([f'the **{t}**' for _, t in docs])}, with page citations."


def build_reply(intent: str, docs: list[tuple[str, str]]) -> IntentReply:
    """Friendly 1-3 sentence reply for a detected intent."""
    covers = _covers(docs)
    try_one = "Try one of the questions below."
    if intent == "greeting":
        text = f"Hello! {covers} {try_one}"
    elif intent == "thanks":
        text = f"You're welcome! Ask me anything else about the documents. {try_one}"
    elif intent == "goodbye":
        text = "Goodbye, and thanks for stopping by! Come back any time you have a question about the documents."
    elif intent == "help":
        text = (f"I'm a document Q&A assistant. {covers} "
                "Ask a specific question in plain English and I'll answer only from the documents, "
                "or say so if the answer isn't there.")
    elif intent == "list_documents":
        if not docs:
            text = "No documents are indexed yet."
        else:
            n = len(docs)
            lines = "\n".join(f"- **{t}** (`{name}`)" if t != name else f"- `{name}`" for name, t in docs)
            text = f"I have {n} document{'s' if n != 1 else ''} indexed:\n\n{lines}"
    elif intent == "too_short":
        text = f"I didn't quite catch a question there. {covers} {try_one}"
    elif intent == "prompt_injection":
        text = (f"I can't change my instructions or share them. {covers} {try_one}")
    else:
        raise ValueError(f"Unknown intent: {intent}")
    return IntentReply(intent=intent, answer=text, suggestions=list(SUGGESTIONS))


def handle(message: str | None, index_dir: Path = INDEX_DIR, data_dir: Path = DATA_DIR) -> IntentReply | None:
    """Detect a small-talk intent and build its reply; None means 'answer from documents'."""
    intent = detect_intent(message)
    if intent is None:
        return None
    return build_reply(intent, document_titles(index_dir, data_dir))
