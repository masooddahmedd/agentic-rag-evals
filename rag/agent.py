"""The agent: plan searches, retrieve, judge relevance, retry with new queries, answer with citations.

Built as a LangGraph state machine rather than free-form tool calling, so a small local model
behaves the same as a large API model: every LLM decision is a small JSON answer. Relevance is
judged by a cross-encoder, not the LLM. In testing, a 3B model marked the right CORS chunk
irrelevant and then answered from the wrong page.

    plan -> retrieve -> grade --(nothing relevant, tries left)--> rewrite -> retrieve
                              \\-> answer
"""
from __future__ import annotations

import os
import re
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .config import settings
from .llm import LLM, parse_json
from .rerank import rerank
from .retrieve import Hit, hybrid_search, rrf

MAX_REWRITES = 1
POOL = 20
# cross-encoder score a chunk needs to count as relevant (calibrated on the eval set, see README)
RELEVANT_SCORE = float(os.environ.get("RELEVANT_SCORE", "0.0"))

PLAN_SYS = (
    "You plan searches over the FastAPI documentation. Turn the user's question into 1 to 3 short "
    "search queries, one per thing that needs looking up. Use FastAPI terms and feature names. "
    'Reply only with JSON: {"queries": ["..."]}'
)
REWRITE_SYS = (
    "Earlier searches of the FastAPI docs found nothing useful. Write 1 or 2 different search "
    "queries: other wording, the FastAPI feature name, or a more general topic. "
    'Reply only with JSON: {"queries": ["..."]}'
)
ANSWER_SYS = (
    "You answer questions about FastAPI using only the numbered sources.\n"
    "Rules:\n"
    "- Every sentence that states how to do something ends with the source it came from, like [1] "
    "or [2][3]. An answer without [n] markers is wrong.\n"
    "- Code must follow a code example in the sources. Copy it and trim it; do not invent other "
    "classes, parameters or patterns.\n"
    "- If the sources do not answer the question, say you couldn't find it in the FastAPI docs and "
    "don't guess.\n"
    "- Be concise: a few sentences and at most one code block.\n"
    "Example: Add CORSMiddleware with the allowed origins [1].\n```python\napp.add_middleware(CORSMiddleware, "
    "allow_origins=origins)\n```\nThe browser then gets the CORS headers it needs [1][2]."
)


class State(TypedDict, total=False):
    question: str
    queries: list[str]
    tried: list[str]
    hits: list[Hit]
    relevant: list[Hit]
    rewrites: int
    answer: str
    citations: list[dict]
    trace: list[dict]


def _trace(state: State, node: str, started: float, call=None, **detail) -> list[dict]:
    step = {"node": node, "ms": round((time.perf_counter() - started) * 1000), **detail}
    if call is not None:
        step["prompt_tokens"], step["completion_tokens"] = call.prompt_tokens, call.completion_tokens
    return [*state.get("trace", []), step]


def _numbered(hits: list[Hit], chars: int | None) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        body = h.content if chars is None else h.content[:chars]
        parts.append(f"[{i}] {h.title} > {h.section}\n{body}")
    return "\n\n".join(parts)


def extract_citations(answer: str, sources: list[Hit]) -> tuple[str, list[dict]]:
    """Keep only [n] markers that point at a real source; list the cited sources in order."""
    used: list[int] = []

    def keep(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= len(sources):
            if n not in used:
                used.append(n)
            return m.group(0)
        return ""
    cleaned = re.sub(r"\[(\d+)\]", keep, answer)
    # a marker inside a code block breaks the code; it still counts as a citation, but is removed there
    cleaned = re.sub(r"```.*?```", lambda m: re.sub(r"[ \t]*\[\d+\]", "", m.group(0)), cleaned, flags=re.S)
    cites =[{"n": n, "doc_path": sources[n - 1].doc_path, "url": sources[n - 1].url,
              "section": sources[n - 1].section, "chunk_id": sources[n - 1].id} for n in sorted(used)]
    return cleaned, cites


class Agent:
    def __init__(self, conn, llm: LLM | None = None, top_k: int | None = None):
        self.conn = conn
        self.llm = llm or LLM()
        self.top_k = top_k or settings.top_k
        self.graph = self._build()

    # ---- nodes ------------------------------------------------------------------
    def plan(self, s: State) -> State:
        t = time.perf_counter()
        call = self.llm.chat(PLAN_SYS, s["question"], json_mode=True, max_tokens=120)
        queries = [q for q in parse_json(call.text).get("queries", []) if isinstance(q, str) and q.strip()][:3]
        # Never skip the search on the planner's say-so: small models misjudge what is on topic.
        # Whether anything relevant came back is decided by the grader instead.
        queries = queries or [s["question"]]
        return {"queries": queries, "tried": list(queries), "rewrites": 0,
                "trace": _trace(s, "plan", t, call, queries=queries)}

    def retrieve(self, s: State) -> State:
        t = time.perf_counter()
        lists = [hybrid_search(self.conn, q, k=POOL) for q in s["queries"]]
        lists.append(hybrid_search(self.conn, s["question"], k=POOL))
        hits = rrf(lists, limit=POOL)
        return {"hits": hits, "trace": _trace(s, "retrieve", t, candidates=len(hits))}

    def grade(self, s: State) -> State:
        t = time.perf_counter()
        # Score each chunk against the question and against the planner's queries, keep the best.
        # "Let my React app call my API" doesn't mention CORS; the planner's "enable CORS" query does.
        best: dict[int, float] = {}
        for text in [s["question"], *s.get("queries", [])]:
            for h in rerank(text, s["hits"]):
                best[h.id] = max(best.get(h.id, float("-inf")), h.score)
        for h in s["hits"]:
            h.score = best[h.id]
        ranked = sorted(s["hits"], key=lambda h: h.score, reverse=True)
        relevant = [h for h in ranked if h.score >= RELEVANT_SCORE][: self.top_k]
        return {"relevant": relevant,
                "trace": _trace(s, "grade", t, relevant=len(relevant),
                                top=[f"{h.doc_path}#{h.section} ({h.score:.1f})" for h in ranked[:3]])}

    def rewrite(self, s: State) -> State:
        t = time.perf_counter()
        user = f"Question: {s['question']}\nAlready tried: {s.get('tried', [])}"
        call = self.llm.chat(REWRITE_SYS, user, json_mode=True, max_tokens=100)
        queries = [q for q in parse_json(call.text).get("queries", []) if isinstance(q, str) and q.strip()][:2]
        queries = queries or [s["question"]]
        return {"queries": queries, "tried": [*s.get("tried", []), *queries], "rewrites": s.get("rewrites", 0) + 1,
                "trace": _trace(s, "rewrite", t, call, queries=queries)}

    def answer(self, s: State) -> State:
        t = time.perf_counter()
        sources = s.get("relevant", [])
        if not sources:
            text = "I couldn't find that in the FastAPI docs."
            return {"answer": text, "citations": [], "trace": _trace(s, "answer", t, sources=0)}
        user = f"Sources:\n\n{_numbered(sources, None)}\n\nQuestion: {s['question']}"
        call = self.llm.chat(ANSWER_SYS, user, max_tokens=500, temperature=0.2)
        text, cites = extract_citations(call.text.strip(), sources)
        trace = _trace(s, "answer", t, call, sources=len(sources))
        if not cites and "couldn't find" not in text.lower():
            # Small models sometimes drop the [n] markers. One retry that only adds them back.
            t2 = time.perf_counter()
            fix = self.llm.chat(ANSWER_SYS, f"{user}\n\nYour answer:\n{text}\n\nRewrite it with the same content, "
                                "adding [n] markers after each sentence that uses a source.", max_tokens=500,
                                temperature=0.0)
            text2, cites2 = extract_citations(fix.text.strip(), sources)
            if cites2:
                text, cites = text2, cites2
            trace = _trace({"trace": trace}, "cite_retry", t2, fix, fixed=bool(cites2))
        return {"answer": text, "citations": cites, "relevant": sources, "trace": trace}

    # ---- graph ------------------------------------------------------------------
    def _after_grade(self, s: State) -> str:
        if not s.get("relevant") and s.get("rewrites", 0) < MAX_REWRITES:
            return "rewrite"
        return "answer"

    def _build(self):
        g = StateGraph(State)
        for name in ("plan", "retrieve", "grade", "rewrite", "answer"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("plan")
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._after_grade, {"rewrite": "rewrite", "answer": "answer"})
        g.add_edge("rewrite", "retrieve")
        g.add_edge("answer", END)
        return g.compile()

    def run(self, question: str) -> State:
        return self.graph.invoke({"question": question, "trace": []})

    def stream(self, question: str):
        """Yields (node, state update) as the graph runs, for the UI's live trace."""
        for update in self.graph.stream({"question": question, "trace": []}, stream_mode="updates"):
            for node, delta in update.items():
                yield node, delta


class PlainRAG:
    """Baseline: one hybrid search on the raw question, answer from the top chunks. No LLM decisions."""

    def __init__(self, conn, llm: LLM | None = None, top_k: int | None = None):
        self.conn, self.llm, self.top_k = conn, llm or LLM(), top_k or settings.top_k

    def run(self, question: str) -> State:
        t = time.perf_counter()
        hits = hybrid_search(self.conn, question, k=self.top_k)
        s: State = {"question": question, "trace": [], "relevant": hits, "queries": [question]}
        s["trace"] = _trace(s, "retrieve", t)
        out = Agent.answer(self, s)   # same answer prompt as the agent
        return {**s, **out}
