"""STUB – contract only. Agent C replaces this file. See TASKS.md."""
from __future__ import annotations

from dataclasses import dataclass

from src.retrieve import TOP_K, RetrievedChunk, retrieve

REFUSAL_MESSAGE = "I cannot find this in the provided documents."


class LLMError(RuntimeError):
    """User-readable LLM failure."""


@dataclass
class AnswerResult:
    answer: str
    cited_chunks: list[RetrievedChunk]
    retrieved: list[RetrievedChunk]
    valid_citations: list[tuple[str, int]]
    invalid_citations: list[tuple[str, int]]
    refused: bool
    error: str | None = None


def generate(prompt: str) -> str:
    raise NotImplementedError


def answer_question(question: str, k: int = TOP_K, retrieve_fn=retrieve, llm_fn=generate) -> AnswerResult:
    raise NotImplementedError
