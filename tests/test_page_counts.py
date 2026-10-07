"""count_pdf_page_chars: per-page counts and an image-only flag, never the document's text.

The case, 2026-10-06: a seat could not record the coverage ledger's page_chars for a large document,
because the only route was reading the whole text back through Dragon's get_document_content, which
spilled to a file the seat may only Read. This tool counts on the server. PyMuPDF is not installed
in CI, so the counting runs over fake pages; the Dockerfile checks the real PyMuPDF at build time.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import page_counts  # noqa: E402
from strict_tool_arguments import strict_tool_arguments  # noqa: E402

SECRET = "CLIENT-ONLY-TEXT-must-never-come-back"


class Page:
    def __init__(self, text):
        self.text = text

    def get_text(self, kind):
        assert kind == "text"
        return self.text


class Doc(list):
    closed = False

    def close(self):
        self.closed = True


def _doc():
    # A text page, a scanned sign ("SIGN", the 2026-10-06 survey), a blank sheet, and a page at
    # the thinnest text density measured (229 characters).
    return Doc([Page(SECRET + " " + "x" * 2000), Page("  SIGN \n"), Page(""), Page("y" * 229)])


def test_counts_each_page_and_flags_the_image_only_ones():
    r = page_counts.page_counts(_doc())
    assert r["page_count"] == 4
    assert r["page_chars"] == {"1": len(SECRET) + 2001, "2": 4, "3": 0, "4": 229}
    assert r["image_only_pages"] == [2, 3]
    assert [p["image_only"] for p in r["pages"]] == [False, True, True, False]
    assert r["image_only_chars_per_page_floor"] == 100


def test_no_text_comes_back():
    assert SECRET not in json.dumps(page_counts.page_counts(_doc()))
    assert "SIGN" not in json.dumps(page_counts.page_counts(_doc()))


def _server(doc):
    mcp = FastMCP("page-counts-test")
    page_counts.register_page_counts(mcp, fetch=lambda url: b"%PDF-fake", opener=lambda data: doc)
    return mcp


def _payload(result):
    """FastMCP.call_tool's return across SDK shapes: the dict the tool returned."""
    if isinstance(result, tuple):
        result = result[1] if isinstance(result[1], dict) else result[0]
    if isinstance(result, dict):
        return result.get("result", result)
    return json.loads(result[0].text)


def test_the_tool_returns_counts_only_and_closes_the_document():
    doc = _doc()
    out = asyncio.run(_server(doc).call_tool("count_pdf_page_chars", {"pdf_url": "https://x.test/a.pdf"}))
    payload = _payload(out)
    assert payload["image_only_pages"] == [2, 3]
    assert SECRET not in json.dumps(payload)
    assert doc.closed


def test_the_tool_refuses_an_undeclared_argument_once_wrapped():
    mcp = _server(_doc())
    strict_tool_arguments(mcp)
    with pytest.raises(ToolError, match="pages"):
        asyncio.run(mcp.call_tool("count_pdf_page_chars", {"pdf_url": "https://x.test/a.pdf", "pages": [1]}))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "/app/auth_app.py", "ftp://x.test/a.pdf"])
def test_only_http_urls_are_fetched(url):
    with pytest.raises(ValueError, match="http"):
        page_counts.fetch_pdf(url)
