"use client";

import { FormEvent, ReactNode, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8800";

type Step = { node: string; ms: number; [k: string]: unknown };
type Citation = { n: number; url: string; doc_path: string; section: string };
type Result = { answer: string; citations: Citation[]; total_ms: number };

const EXAMPLES = [
  "How do I let my React app on localhost:3000 call my API?",
  "How do I load an ML model once at startup and free it on shutdown?",
  "How do I connect FastAPI to a Cassandra cluster?",
];

function describe(s: Step): string {
  if (s.queries) return (s.queries as string[]).join(" · ");
  if (s.node === "retrieve") return `${s.candidates ?? "?"} candidate chunks`;
  if (s.node === "grade") return `${s.relevant} relevant${s.top ? " · best: " + (s.top as string[])[0] : ""}`;
  if (s.node === "answer") return `${s.sources ?? 0} sources`;
  return "";
}

// Render the answer: fenced code as <pre>, [n] markers as links to the cited page.
function render(answer: string, cites: Citation[]): ReactNode[] {
  const byN = new Map(cites.map(c => [c.n, c]));
  return answer.split(/(```[\s\S]*?```)/g).map((part, i) => {
    if (part.startsWith("```")) {
      return <pre key={i}><code>{part.replace(/^```\w*\n?/, "").replace(/```$/, "")}</code></pre>;
    }
    return (
      <span key={i}>
        {part.split(/(\[\d+\]|`[^`\n]+`)/g).map((t, j) => {
          if (t.startsWith("`") && t.endsWith("`") && t.length > 2) return <code key={j}>{t.slice(1, -1)}</code>;
          const m = t.match(/^\[(\d+)\]$/);
          const c = m ? byN.get(Number(m[1])) : undefined;
          return c ? <a key={j} className="cite" href={c.url} target="_blank" rel="noreferrer">{t}</a> : t;
        })}
      </span>
    );
  });
}

export default function Page() {
  const [q, setQ] = useState(EXAMPLES[0]);
  const [mode, setMode] = useState("agent");
  const [steps, setSteps] = useState<Step[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function ask(e: FormEvent) {
    e.preventDefault();
    setBusy(true); setSteps([]); setResult(null); setError("");
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, mode }),
      });
      if (!res.ok || !res.body) throw new Error(`API returned ${res.status}`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const events = buf.split("\n\n");
        buf = events.pop() || "";
        for (const ev of events) {
          const type = ev.match(/^event: (\w+)/m)?.[1];
          const data = JSON.parse(ev.match(/^data: (.*)$/m)?.[1] || "{}");
          if (type === "step") setSteps(s => [...s, data]);
          if (type === "answer") setResult(data);
        }
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <h1>FastAPI docs agent</h1>
      <p className="sub">Plans searches, checks what it found, retries when nothing fits, and answers only from cited docs.</p>
      <form onSubmit={ask}>
        <input type="text" value={q} onChange={e => setQ(e.target.value)} aria-label="Question" />
        <select value={mode} onChange={e => setMode(e.target.value)} aria-label="Mode">
          <option value="agent">Agent</option>
          <option value="plain">Plain RAG</option>
        </select>
        <button disabled={busy}>{busy ? "Thinking…" : "Ask"}</button>
      </form>
      <p className="muted" style={{ fontSize: 13 }}>
        Try: {EXAMPLES.slice(1).map((x, i) => (
          <a key={i} href="#" className="cite" style={{ marginRight: 10 }} onClick={e => { e.preventDefault(); setQ(x); }}>{x}</a>
        ))}
      </p>
      <div className="grid">
        <section className="card">
          <h2>Answer</h2>
          {error && <p>{error}</p>}
          {!result && !error && <p className="muted">{busy ? "Working…" : "Ask something about FastAPI."}</p>}
          {result && (
            <>
              <div className="answer">{render(result.answer, result.citations)}</div>
              {result.citations.length > 0 && (
                <ol className="sources">
                  {result.citations.map(c => (
                    <li key={c.n} value={c.n}><a href={c.url} target="_blank" rel="noreferrer">{c.doc_path}</a> · {c.section}</li>
                  ))}
                </ol>
              )}
            </>
          )}
        </section>
        <aside className="card">
          <h2>Steps</h2>
          {steps.length === 0 && <p className="muted">–</p>}
          {steps.map((s, i) => (
            <div className="step" key={i}>
              <span className="ms">{(s.ms / 1000).toFixed(1)} s</span>
              <b>{s.node}</b>
              <div className="detail">{describe(s)}</div>
            </div>
          ))}
          {result && <div className="step"><b>Total</b><span className="ms">{(result.total_ms / 1000).toFixed(1)} s</span></div>}
        </aside>
      </div>
    </main>
  );
}
