"""Local embeddings with fastembed (ONNX, CPU). bge-small needs a prefix on queries."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding
    return TextEmbedding(settings.embed_model)


def embed_passages(texts: list[str], batch_size: int = 64) -> list[np.ndarray]:
    return list(_model().embed(texts, batch_size=batch_size))


def embed_query(text: str) -> np.ndarray:
    prefix = QUERY_PREFIX if "bge" in settings.embed_model.lower() else ""
    return next(iter(_model().embed([prefix + text])))
