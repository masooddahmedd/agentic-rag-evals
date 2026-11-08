from pathlib import Path

from rag.agent import extract_citations
from rag.ingest import _blocks, chunk_page, page_url, split_page
from rag.llm import parse_json
from rag.retrieve import Hit, rrf


def hit(i, path="a.md"):
    return Hit(i, path, "T", "S", "u", "c")


def test_split_page_sections_and_title():
    text = "# Title\n\nintro\n\n## One\n\nbody1\n\n### Sub\n\nbody2\n\n## Two\n\nbody3"
    title, sections = split_page(text)
    assert title == "Title"
    assert [s for s, _ in sections] == ["Introduction", "One", "One > Sub", "Two"]


def test_headings_inside_code_are_not_sections():
    text = "# T\n\n## Real\n\n```python\n# not a heading\nx = 1\n```"
    _, sections = split_page(text)
    assert len(sections) == 1 and "# not a heading" in sections[0][1]


def test_blocks_keep_code_fences_whole():
    body = "para one\n\n```python\na = 1\n\nb = 2\n```\n\npara two"
    blocks = _blocks(body)
    assert len(blocks) == 3 and "b = 2" in blocks[1]


def test_long_sections_are_split():
    para = " ".join(["word"] * 200)
    chunks = chunk_page("tutorial/x.md", f"# X\n\n## S\n\n{para}\n\n{para}")
    assert len(chunks) == 2 and all(c.section == "S" for c in chunks)


def test_page_url():
    assert page_url("tutorial/cors.md") == "https://fastapi.tiangolo.com/tutorial/cors/"
    assert page_url("tutorial/security/index.md") == "https://fastapi.tiangolo.com/tutorial/security/"
    assert page_url("index.md") == "https://fastapi.tiangolo.com/"


def test_rrf_rewards_agreement():
    a = [hit(1), hit(2), hit(3)]
    b = [hit(3), hit(4), hit(1)]
    fused = [h.id for h in rrf([a, b], limit=4)]
    assert fused[0] in (1, 3) and set(fused) == {1, 2, 3, 4}


def test_citations_out_of_range_are_dropped():
    sources = [hit(10, "x.md"), hit(11, "y.md")]
    text, cites = extract_citations("Use CORS [1][7]. Also this [2].", sources)
    assert "[7]" not in text
    assert [c["doc_path"] for c in cites] == ["x.md", "y.md"]


def test_parse_json_tolerates_wrapping():
    assert parse_json('Sure! ```json\n{"queries": ["cors"]}\n```') == {"queries": ["cors"]}
    assert parse_json("no json here") == {}


def test_markers_inside_code_are_removed_but_still_count():
    sources = [hit(10, "cors.md")]
    text, cites = extract_citations("Add it [1].\n```python\nallow_origins=origins, [1]\n```", sources)
    assert "origins, [1]" not in text and "origins," in text
    assert [c["doc_path"] for c in cites] == ["cors.md"]
