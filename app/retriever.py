"""Chunking, embeddings, and Chroma search."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str  # path relative to the repo root, e.g. "data/kb/refunds.md"
    text: str


@lru_cache(maxsize=1)
def _model():
    # Imported here so that code which never embeds (e.g. unit tests of the
    # LLM client) does not pay the torch import cost.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Return one unit-length vector per text."""
    return _model().encode(texts, normalize_embeddings=True).tolist()


def split_markdown(text: str, source: str) -> list[Chunk]:
    """One chunk per '## ' section, each prefixed with the document title."""
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else Path(source).stem
    sections = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    chunks = []
    for index, section in enumerate(sections):
        heading, _, body = section.partition("\n")
        chunks.append(
            Chunk(
                id=f"{source}#{index}",
                source=source,
                text=f"{title} - {heading.strip()}\n{body.strip()}",
            )
        )
    return chunks


def load_chunks(kb_dirs: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for kb_dir in kb_dirs:
        directory = Path(kb_dir)
        if not directory.is_absolute():
            directory = ROOT / directory
        for path in sorted(directory.glob("*.md")):
            try:
                source = path.relative_to(ROOT).as_posix()
            except ValueError:
                source = path.as_posix()
            chunks.extend(split_markdown(path.read_text(encoding="utf-8"), source))
    if not chunks:
        raise ValueError(f"No markdown documents found in {kb_dirs}")
    return chunks


class Retriever:
    def __init__(self, kb_dirs: list[str]):
        import chromadb

        self.chunks = {chunk.id: chunk for chunk in load_chunks(kb_dirs)}
        # In-process clients share one store, so each document set gets its
        # own collection name.
        name = "kb-" + hashlib.sha256("|".join(sorted(self.chunks)).encode()).hexdigest()[:16]
        client = chromadb.EphemeralClient()
        self.collection = client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )
        if self.collection.count() == 0:
            ordered = list(self.chunks.values())
            self.collection.add(
                ids=[c.id for c in ordered],
                documents=[c.text for c in ordered],
                embeddings=embed([c.text for c in ordered]),
            )

    def search(self, question: str, k: int = TOP_K) -> list[Chunk]:
        result = self.collection.query(query_embeddings=embed([question]), n_results=k)
        return [self.chunks[chunk_id] for chunk_id in result["ids"][0]]
