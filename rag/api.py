"""HTTP API. POST /ask streams the agent's steps as server-sent events, then the answer.

    uvicorn rag.api:app --port 8800
"""
from __future__ import annotations

import json
import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import Agent, PlainRAG
from .db import connect

app = FastAPI(title="Agentic RAG over the FastAPI docs")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
                   allow_methods=["*"], allow_headers=["*"])
_conn = None


def conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = connect()
    return _conn


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = Field("agent", pattern="^(agent|plain)$")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _public(step: dict) -> dict:
    return {k: v for k, v in step.items() if k != "node"} | {"node": step["node"]}


@app.get("/health")
def health():
    n = conn().execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"ok": True, "chunks": n}


@app.post("/ask")
def ask(body: Ask):
    def events():
        t = time.perf_counter()
        if body.mode == "plain":
            state = PlainRAG(conn()).run(body.question)
            for step in state.get("trace", []):
                yield _sse("step", _public(step))
        else:
            state, sent = {}, 0
            for node, delta in Agent(conn()).stream(body.question):
                state.update(delta)
                trace = delta.get("trace") or []
                for step in trace[sent:]:          # a node can add more than one step (answer + cite_retry)
                    yield _sse("step", _public(step))
                sent = max(sent, len(trace))
        yield _sse("answer", {"answer": state.get("answer", ""), "citations": state.get("citations", []),
                              "total_ms": round((time.perf_counter() - t) * 1000)})
    return StreamingResponse(events(), media_type="text/event-stream")
