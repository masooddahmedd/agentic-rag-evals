"""Answer quality for the agent and the plain-RAG baseline.

Two passes, so the judge can be swapped without regenerating answers:

    python -m evals.agent_eval answer --mode agent plain      # runs the systems, saves answers + cited sources
    python -m evals.agent_eval grade --mode agent plain       # judge model grades the saved answers

Per question: facts covered (judge checks each expected fact), whether the answer cites anything,
whether the cited sources back every claim (judge lists unsupported claims first), citation
precision (cited page is a gold page), correct refusal on questions the docs can't answer,
latency and tokens.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from rag.config import settings
from rag.llm import LLM, parse_json

HERE = Path(__file__).parent
RUNS = HERE.parent / "runs"

REFUSAL = ("couldn't find", "could not find", "not in the fastapi docs", "doesn't cover", "does not cover",
           "no information", "not mentioned")

FACTS_SYS = (
    "You grade an answer about FastAPI. For each expected fact, say whether the answer states it or "
    "something equivalent (code counts). Reply only with JSON: {\"covered\": [true or false per fact]}"
)
FAITH_SYS = (
    "You check whether an answer about FastAPI is backed by its sources. First list every claim or code "
    "pattern in the answer that the sources do not support. Renaming variables or trimming an example is "
    "fine; a different API, parameter or behaviour is not. Then give the verdict: 'yes' if nothing "
    "important is unsupported, 'partly' if only minor details are added, 'no' if a key instruction or code "
    'pattern is not in the sources. Reply only with JSON: {"unsupported": ["..."], "supported": "yes|partly|no"}'
)


def load_questions() -> list[dict]:
    return [json.loads(l) for l in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def refused(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in REFUSAL)


# ------------------------------------------------------------------ pass 1: answers

def run_answers(modes: list[str], limit: int) -> None:
    from rag.agent import Agent, PlainRAG
    from rag.db import connect
    conn, llm = connect(), LLM()
    questions = load_questions()[: limit or None]
    RUNS.mkdir(exist_ok=True)
    for mode in modes:
        runner = Agent(conn, llm) if mode == "agent" else PlainRAG(conn, llm)
        with open(RUNS / f"answers_{mode}.jsonl", "w", encoding="utf-8") as out:
            for q in questions:
                t = time.perf_counter()
                state = runner.run(q["question"])
                ms = round((time.perf_counter() - t) * 1000)
                trace = state.get("trace", [])
                cited_ids = {c["chunk_id"] for c in state.get("citations", [])}
                row = {
                    "id": q["id"], "type": q["type"], "gold": q["gold"], "ms": ms,
                    "tokens": sum(s.get("prompt_tokens", 0) + s.get("completion_tokens", 0) for s in trace),
                    "llm_calls": sum("prompt_tokens" in s for s in trace),
                    "path": [s["node"] for s in trace],
                    "answer": state.get("answer", ""), "citations": state.get("citations", []),
                    "cited_sources": [h.content for h in state.get("relevant", []) if h.id in cited_ids],
                    "context_pages": sorted({h.doc_path for h in state.get("relevant", [])}),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(mode, q["id"], f"{ms / 1000:.1f}s", flush=True)


# -------------------------------------------------------------------- pass 2: grade

def grade_row(judge: LLM, q: dict, row: dict) -> dict:
    answer = row["answer"]
    rec: dict = {"refused": refused(answer)}
    if not q["gold"]:
        rec["correct_refusal"] = rec["refused"]
        return rec
    facts = q["facts"]
    call = judge.chat(FACTS_SYS, json.dumps({"question": q["question"], "expected_facts": facts, "answer": answer}),
                      json_mode=True, max_tokens=150)
    covered = parse_json(call.text).get("covered", [])
    if len(covered) != len(facts):        # judge broke format: fall back to substring match
        covered = [f.lower() in answer.lower() for f in facts]
    rec["facts"] = round(sum(bool(c) for c in covered) / len(facts), 3)
    rec["cited"] = bool(row["citations"])
    rec["gold_in_context"] = any(p in q["gold"] for p in row["context_pages"])
    rec["citation_precision"] = (round(sum(c["doc_path"] in q["gold"] for c in row["citations"]) / len(row["citations"]), 3)
                                 if row["citations"] else None)
    if rec["refused"] or not row["cited_sources"]:
        rec["supported"] = "n/a"
    else:
        ctx = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(row["cited_sources"], 1))
        data = parse_json(judge.chat(FAITH_SYS, f"Sources:\n{ctx}\n\nAnswer:\n{answer}", json_mode=True,
                                     max_tokens=400).text)
        rec["supported"] = data.get("supported", "no")
        rec["unsupported_claims"] = data.get("unsupported", [])
    return rec


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["gold"]]
    unans = [r for r in rows if not r["gold"]]
    judged = [r for r in ans if r.get("supported") in ("yes", "partly", "no")]
    prec = [r["citation_precision"] for r in ans if r.get("citation_precision") is not None]
    lat = sorted(r["ms"] for r in rows)
    share = lambda xs, v: round(sum(r["supported"] == v for r in xs) / max(1, len(xs)), 3)
    return {
        "questions": len(rows),
        "facts_covered": round(statistics.mean(r["facts"] for r in ans), 3),
        "answers_with_citations": round(sum(r["cited"] for r in ans) / len(ans), 3),
        "cited_answers_supported": share(judged, "yes"),
        "cited_answers_partly": share(judged, "partly"),
        "cited_answers_unsupported": share(judged, "no"),
        "citation_precision": round(statistics.mean(prec), 3) if prec else None,
        "gold_page_in_context": round(sum(r["gold_in_context"] for r in ans) / len(ans), 3),
        "wrongly_refused": sum(r["refused"] for r in ans),
        "correct_refusals": f"{sum(r['correct_refusal'] for r in unans)}/{len(unans)}",
        "latency_p50_s": round(statistics.median(lat) / 1000, 1),
        "latency_p90_s": round(lat[int(0.9 * (len(lat) - 1))] / 1000, 1),
        "tokens_per_question": round(statistics.mean(r["tokens"] for r in rows)),
        "llm_calls_per_question": round(statistics.mean(r["llm_calls"] for r in rows), 2),
    }


def run_grades(modes: list[str]) -> None:
    judge = LLM(model=settings.judge_model)
    questions = {q["id"]: q for q in load_questions()}
    summary = {}
    for mode in modes:
        rows = [json.loads(l) for l in open(RUNS / f"answers_{mode}.jsonl", encoding="utf-8")]
        graded = []
        for row in rows:
            g = {**row, **grade_row(judge, questions[row["id"]], row)}
            graded.append(g)
            print(mode, row["id"], {k: g.get(k) for k in ("facts", "cited", "supported", "refused")}, flush=True)
        with open(RUNS / f"graded_{mode}.jsonl", "w", encoding="utf-8") as f:
            for g in graded:
                f.write(json.dumps({k: v for k, v in g.items() if k != "cited_sources"}) + "\n")
        summary[mode] = summarize(graded)
        print(mode, json.dumps(summary[mode]), flush=True)
    summary["models"] = {"answer": settings.llm_model, "judge": settings.judge_model,
                         "embeddings": settings.embed_model}
    (RUNS / "agent_eval.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["answer", "grade", "both"])
    ap.add_argument("--mode", nargs="+", default=["agent", "plain"], choices=["agent", "plain"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.step in ("answer", "both"):
        run_answers(a.mode, a.limit)
    if a.step in ("grade", "both"):
        run_grades(a.mode)


if __name__ == "__main__":
    main()
