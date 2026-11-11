"""Cosine similarity between two answers.

Uses the same sentence-transformers model as the retriever, so there is one
embedding model to pin and one to download.
"""
from __future__ import annotations

from app.retriever import embed


def cosine_similarity(a: str, b: str) -> float:
    vec_a, vec_b = embed([a, b])
    # embed() returns unit-length vectors, so the dot product is the cosine.
    return sum(x * y for x, y in zip(vec_a, vec_b))
