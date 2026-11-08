"""Load the FastAPI docs, chunk them by heading and store chunks with embeddings.

    python -m rag.ingest --docs data/fastapi-src --reset

The docs pull code samples in with mkdocs include syntax ({* ../../docs_src/x.py *}). The
code is often the answer to a "how do I" question, so includes are replaced with the code.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from .config import settings

SKIP = {
    "release-notes.md", "external-links.md", "fastapi-people.md", "newsletter.md",
    "help-fastapi.md", "contributing.md", "management.md", "management-tasks.md",
    "project-generation.md", "benchmarks.md", "reference", "_llm-test.md", "agent-banner.md",
    "translation-banner.md", "translations.md", "about", "learn", "resources", "ai-development.md",
    "deployment/fastapicloud.md",
}
MAX_WORDS = 280

INCLUDE_STAR = re.compile(r"\{\*\s*([^\s*]+)[^*]*\*\}")
INCLUDE_BANG = re.compile(r"\{!>?\s*([^\s!]+)\s*!\}")
ADMONITION = re.compile(r"^///\s*(\w+)?(?:\s*\|\s*(.*))?$")
HTML_TAG = re.compile(r"<[^>]+>")
HEADING = re.compile(r"^(#{1,3})\s+(.*?)\s*(\{[^}]*\})?\s*$")


@dataclass
class Chunk:
    doc_path: str
    title: str
    section: str
    url: str
    content: str

    def embed_text(self) -> str:
        return f"{self.title} > {self.section}\n{self.content}"


def page_url(rel: str) -> str:
    path = rel[:-3]
    if path.endswith("index"):
        path = path[: -len("index")]
    return f"https://fastapi.tiangolo.com/{path}".rstrip("/") + "/"


def _include(match: re.Match, docs_root: Path, md_file: Path) -> str:
    target = (md_file.parent / match.group(1)).resolve()
    if not target.exists():
        # paths are relative to docs/en/docs; fall back to the repo root
        alt = docs_root.parents[2] / match.group(1).replace("../", "")
        target = alt if alt.exists() else target
    if not target.exists():
        return ""
    lang = "python" if target.suffix == ".py" else ""
    return f"\n```{lang}\n{target.read_text(encoding='utf-8').strip()}\n```\n"


def clean(text: str, docs_root: Path, md_file: Path) -> str:
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)        # front matter
    text = INCLUDE_STAR.sub(lambda m: _include(m, docs_root, md_file), text)
    text = INCLUDE_BANG.sub(lambda m: _include(m, docs_root, md_file).strip().strip("`").strip(), text)
    lines = []
    for line in text.splitlines():
        m = ADMONITION.match(line.strip())
        if m:
            label = (m.group(2) or m.group(1) or "").strip()
            if label:
                lines.append(f"{label.capitalize()}:")
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = HTML_TAG.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _blocks(body: str) -> list[str]:
    """Paragraphs, keeping fenced code blocks whole."""
    out, cur, in_code = [], [], False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if cur:
                out.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return out


def split_page(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(page title, [(section path, body)]) split on ## and ### headings."""
    title, sections = "", []
    h2 = h3 = ""
    body: list[str] = []
    in_code = False

    def flush():
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(x for x in (h2, h3) if x) or "Introduction", content))

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m:
            level, name = len(m.group(1)), m.group(2).strip()
            if level == 1:
                title = title or name
                continue
            flush()
            body = []
            if level == 2:
                h2, h3 = name, ""
            else:
                h3 = name
            continue
        body.append(line)
    flush()
    return title, sections


def chunk_page(rel: str, text: str) -> list[Chunk]:
    title, sections = split_page(text)
    title = title or Path(rel).stem.replace("-", " ").title()
    chunks = []
    for section, body in sections:
        cur, words = [], 0
        for block in _blocks(body):
            n = len(block.split())
            if cur and words + n > MAX_WORDS:
                chunks.append(Chunk(rel, title, section, page_url(rel), "\n\n".join(cur)))
                cur, words = [], 0
            cur.append(block)
            words += n
        if cur:
            chunks.append(Chunk(rel, title, section, page_url(rel), "\n\n".join(cur)))
    return chunks


def load_chunks(src: Path) -> list[Chunk]:
    docs_root = src / "docs" / "en" / "docs"
    chunks = []
    for md in sorted(docs_root.rglob("*.md")):
        rel = md.relative_to(docs_root).as_posix()
        if rel in SKIP or rel.split("/")[0] in SKIP or rel.startswith("deployment/fastapicloud"):
            continue
        text = clean(md.read_text(encoding="utf-8"), docs_root, md)
        chunks.extend(chunk_page(rel, text))
    return chunks


def main() -> None:
    from .db import connect, create_schema
    from .embed import embed_passages

    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=settings.docs_dir, help="FastAPI repo checkout (needs docs/ and docs_src/)")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()

    chunks = load_chunks(Path(a.docs))
    print(f"{len(chunks)} chunks from {len({c.doc_path for c in chunks})} pages")
    vectors = embed_passages([c.embed_text() for c in chunks])
    conn = connect()
    create_schema(conn, reset=a.reset)
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (doc_path, title, section, url, content, embedding) VALUES (%s,%s,%s,%s,%s,%s)",
            [(c.doc_path, c.title, c.section, c.url, c.content, v) for c, v in zip(chunks, vectors)])
    print("stored", conn.execute("SELECT count(*) FROM chunks").fetchone()[0])


if __name__ == "__main__":
    main()
