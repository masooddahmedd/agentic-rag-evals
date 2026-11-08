"""Retrieval quality per retriever, no LLM involved. Fast enough to run in CI.

    python -m evals.retrieval_eval --k 5 --min-hybrid-hit 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.db import connect
from rag.retrieve import RETRIEVERS

HERE = Path(__file__).parent


def load_questions() -> list[dict]:
    return [json.loads(l) for l in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(conn, questions: list[dict], k: int) -> dict:
    out = {}
    answerable = [q for q in questions if q["gold"]]
    for name, fn in RETRIEVERS.items():
        hits, rr, misses = 0, 0.0, []
        for q in answerable:
            docs = [h.doc_path for h in fn(conn, q["question"], k)]
            rank = next((i for i, d in enumerate(docs, 1) if d in q["gold"]), None)
            if rank:
                hits += 1
                rr += 1 / rank
            else:
                misses.append(q["id"])
        out[name] = {"hit_at_k": round(hits / len(answerable), 3), "mrr": round(rr / len(answerable), 3),
                     "misses": misses}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-hybrid-hit", type=float, default=0.0, help="fail (exit 1) below this")
    a = ap.parse_args()
    res = evaluate(connect(), load_questions(), a.k)
    print(f"{'retriever':14} hit@{a.k}   MRR")
    for name, r in res.items():
        print(f"{name:14} {r['hit_at_k']:.3f}   {r['mrr']:.3f}   misses: {', '.join(r['misses']) or '-'}")
    (HERE.parent / "runs").mkdir(exist_ok=True)
    (HERE.parent / "runs" / "retrieval.json").write_text(json.dumps(res, indent=2))
    if res["hybrid"]["hit_at_k"] < a.min_hybrid_hit:
        sys.exit(f"hybrid hit@{a.k} {res['hybrid']['hit_at_k']} below {a.min_hybrid_hit}")


if __name__ == "__main__":
    main()
