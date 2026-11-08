"""Cross-encoder reranking (fastembed, CPU). Reads question and chunk together, so it judges
relevance far more consistently than a small LLM asked "is this snippet relevant?"."""
from __future__ import annotations

import os
from functools import lru_cache

from .retrieve import Hit

RERANK_MODEL = os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-12-v2")


@lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(RERANK_MODEL)


def rerank(question: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
    if not hits:
        return []
    docs = [f"{h.title} > {h.section}\n{h.content[:1500]}" for h in hits]
    scores = list(_model().rerank(question, docs))
    for h, s in zip(hits, scores):
        h.score = float(s)
    ordered = sorted(hits, key=lambda h: h.score, reverse=True)
    return ordered[:top_k] if top_k else ordered
