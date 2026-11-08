"""Pick the reranker score threshold for "relevant".

For answerable questions it prints the best score a gold-page chunk gets; for unanswerable ones,
the best score any chunk gets. A good threshold sits between the two groups.
"""
from __future__ import annotations

import json
from pathlib import Path

from rag.db import connect
from rag.retrieve import hybrid_rerank_search

HERE = Path(__file__).parent


def main() -> None:
    conn = connect()
    gold, off = [], []
    for line in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        hits = hybrid_rerank_search(conn, q["question"], k=5)
        if q["gold"]:
            scores = [h.score for h in hits if h.doc_path in q["gold"]]
            gold.append((q["id"], round(max(scores), 2) if scores else None))
        else:
            off.append((q["id"], round(hits[0].score, 2)))
    print("best gold chunk, answerable:", sorted((s for _, s in gold if s is not None)))
    print("gold not in top 5:", [i for i, s in gold if s is None])
    print("best chunk, unanswerable:", off)


if __name__ == "__main__":
    main()
