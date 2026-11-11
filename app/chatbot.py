"""ask(question) -> Answer(text, chunks).

The answer always comes back with the chunks that were retrieved for it,
so tests can compare what the model said with what it was given.

Command line:  python -m app.chatbot "How long do I have to return an item?"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache

from app.llm import LLMClient, get_llm
from app.prompt import build_prompt
from app.retriever import Chunk, Retriever

DEFAULT_KB_DIRS = ("data/kb",)


@dataclass(frozen=True)
class Answer:
    text: str
    chunks: list[Chunk]


class Chatbot:
    def __init__(self, retriever: Retriever, llm: LLMClient):
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str) -> Answer:
        chunks = self.retriever.search(question)
        system, user = build_prompt(question, chunks)
        return Answer(text=self.llm.complete(system, user), chunks=chunks)


def build_chatbot(kb_dirs: list[str] | tuple[str, ...] = DEFAULT_KB_DIRS, llm: LLMClient | None = None) -> Chatbot:
    return Chatbot(Retriever(list(kb_dirs)), llm or get_llm("app"))


@lru_cache(maxsize=1)
def _default_chatbot() -> Chatbot:
    return build_chatbot()


def ask(question: str) -> Answer:
    return _default_chatbot().ask(question)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m app.chatbot "your question"')
    answer = ask(" ".join(sys.argv[1:]))
    print(f"Answer:\n{answer.text}\n")
    print("Chunks:")
    for chunk in answer.chunks:
        print(f"- [{chunk.id}] {chunk.text.splitlines()[0]}")


if __name__ == "__main__":
    main()
