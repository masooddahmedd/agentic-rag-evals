"""Sanity check for the eval set: every gold page exists and every literal fact appears on a gold page.

Facts written as plain phrases (with spaces and no code characters) are judged by the model and
only warned about here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rag.db import connect

HERE = Path(__file__).parent


def main() -> None:
    conn = connect()
    pages = {r[0]: "" for r in conn.execute("SELECT DISTINCT doc_path FROM chunks")}
    for path, content in conn.execute("SELECT doc_path, string_agg(content, ' ') FROM chunks GROUP BY doc_path"):
        pages[path] = content.lower()
    bad = 0
    for line in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        for g in q["gold"]:
            if g not in pages:
                print(f"{q['id']}: gold page missing: {g}")
                bad += 1
        text = " ".join(pages.get(g, "") for g in q["gold"])
        for f in q["facts"]:
            literal = not (" " in f and re.fullmatch(r"[A-Za-z ]+", f))
            if f.lower() not in text:
                print(f"{q['id']}: {'MISSING' if literal else 'check by judge'}: {f!r}")
                bad += literal
    print("ok" if not bad else f"{bad} problems")


if __name__ == "__main__":
    main()
