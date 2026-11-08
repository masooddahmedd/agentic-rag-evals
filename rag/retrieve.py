"""Three ways to find chunks: vector similarity, Postgres full-text, and both fused (RRF)."""
from __future__ import annotations

from dataclasses import dataclass

from .embed import embed_query


@dataclass
class Hit:
    id: int
    doc_path: str
    title: str
    section: str
    url: str
    content: str
    score: float = 0.0


COLS = "id, doc_path, title, section, url, content"


def vector_search(conn, query: str, k: int = 10) -> list[Hit]:
    q = embed_query(query)
    rows = conn.execute(
        f"SELECT {COLS}, 1 - (embedding <=> %s) AS s FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (q, q, k)).fetchall()
    return [Hit(*r) for r in rows]


def keyword_search(conn, query: str, k: int = 10) -> list[Hit]:
    # OR the words together: plain websearch_to_tsquery ANDs them, which is too strict for questions
    rows = conn.execute(
        f"""WITH q AS (SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery AS tq)
            SELECT {COLS}, ts_rank_cd(tsv, q.tq) AS s FROM chunks, q
            WHERE tsv @@ q.tq ORDER BY s DESC LIMIT %s""",
        (query, k)).fetchall()
    return [Hit(*r) for r in rows]


def rrf(result_lists: list[list[Hit]], k: int = 60, limit: int = 10) -> list[Hit]:
    """Reciprocal rank fusion: score = sum of 1 / (k + rank) across lists."""
    scores: dict[int, float] = {}
    hits: dict[int, Hit] = {}
    for results in result_lists:
        for rank, h in enumerate(results, start=1):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (k + rank)
            hits.setdefault(h.id, h)
    ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
    out = []
    for i in ordered:
        h = hits[i]
        h.score = scores[i]
        out.append(h)
    return out


def hybrid_search(conn, query: str, k: int = 10, pool: int = 20) -> list[Hit]:
    return rrf([vector_search(conn, query, pool), keyword_search(conn, query, pool)], limit=k)


def hybrid_rerank_search(conn, query: str, k: int = 10, pool: int = 20) -> list[Hit]:
    from .rerank import rerank
    return rerank(query, hybrid_search(conn, query, k=pool, pool=pool), top_k=k)


RETRIEVERS = {"vector": vector_search, "keyword": keyword_search, "hybrid": hybrid_search,
              "hybrid+rerank": hybrid_rerank_search}
