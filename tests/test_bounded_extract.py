"""bounded_extract.py: a slow extraction must not freeze the instance, and a vector-dense page
must not go through pymupdf4llm's layout analysis.

Measured 2026-10-04 on the service's logs: a sync tool ran on the event loop, so one 215 s
extraction held every other request until Cloud Run cut each at 300 s. The PyMuPDF stack is
not installed in CI, so the second half runs against fake pages that answer only what
bounded_extract asks of them.
"""

import asyncio
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bounded_extract  # noqa: E402
from bounded_extract import MAX_DRAWINGS_PER_PAGE, bound_to_markdown, run_tools_in_threads  # noqa: E402

BLOCK_S = 0.5


def _server():
    mcp = FastMCP("slow-stub")

    @mcp.tool()
    def slow(x: int = 0) -> str:
        time.sleep(BLOCK_S)  # a blocking C call in the real tool
        return "done"

    return mcp


async def _ticks_during(mcp) -> int:
    """How many 50 ms ticks an unrelated coroutine got while the tool ran."""
    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.05)
            ticks += 1

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)  # let the ticker start before the tool is called
    await mcp.call_tool("slow", {})
    stop.set()
    await task
    return ticks


def test_a_sync_tool_freezes_the_loop_unless_it_is_threaded():
    # The control: this is production before the fix. The loop is blocked for the whole call.
    assert asyncio.run(_ticks_during(_server())) <= 1
    mcp = _server()
    assert run_tools_in_threads(mcp) == 1
    assert asyncio.run(_ticks_during(mcp)) >= 5


def test_a_threaded_tool_still_returns_its_result_and_validates_arguments():
    mcp = _server()
    run_tools_in_threads(mcp)
    _, structured = asyncio.run(mcp.call_tool("slow", {"x": 3}))
    assert structured == {"result": "done"}


def test_threading_with_no_tools_registered_is_an_error_not_a_no_op():
    import pytest

    with pytest.raises(RuntimeError, match="no tools registered"):
        run_tools_in_threads(FastMCP("empty"))


class _Page:
    def __init__(self, drawings, text):
        self._drawings, self._text = drawings, text

    def get_cdrawings(self):
        return [None] * self._drawings

    def get_text(self, mode):
        assert mode == "text"
        return self._text


class _Doc:
    def __init__(self, pages):
        self._pages = pages
        self.page_count = len(pages)

    def __getitem__(self, i):
        return self._pages[i]


def _recording_original():
    calls = []

    def original(doc, *args, pages=None, **kwargs):
        calls.append({"pages": None if pages is None else list(pages), "args": args, "kwargs": kwargs})
        return [{"text": f"md{p}"} for p in (range(doc.page_count) if pages is None else pages)]

    return original, calls


def test_pages_under_the_limit_take_the_original_path_untouched():
    original, calls = _recording_original()
    doc = _Doc([_Page(10, "a"), _Page(MAX_DRAWINGS_PER_PAGE, "b")])
    out = bound_to_markdown(original)(doc, pages=[0, 1], page_chunks=True, use_ocr=False)
    assert out == [{"text": "md0"}, {"text": "md1"}]
    assert calls == [{"pages": [0, 1], "args": (), "kwargs": {"page_chunks": True, "use_ocr": False}}]


def test_a_dense_page_is_read_as_plain_text_and_says_so_while_the_rest_keep_layout():
    original, calls = _recording_original()
    doc = _Doc([_Page(5, "a"), _Page(MAX_DRAWINGS_PER_PAGE + 1, "plain sheet text"), _Page(7, "c")])
    out = bound_to_markdown(original)(doc, pages=[0, 1, 2], page_chunks=True, use_ocr=False)
    assert [c["text"] for c in out][0] == "md0"
    assert out[2] == {"text": "md2"}
    assert out[1]["text"].startswith("[page 2: ")
    assert f"{MAX_DRAWINGS_PER_PAGE + 1} vector drawings" in out[1]["text"]
    assert out[1]["text"].endswith("plain sheet text")
    # The dense page never reached the original, and the others went in one call, in order.
    assert [c["pages"] for c in calls] == [[0, 2]]


def test_a_document_of_only_dense_pages_never_calls_the_original():
    original, calls = _recording_original()
    doc = _Doc([_Page(MAX_DRAWINGS_PER_PAGE + 5, "x"), _Page(MAX_DRAWINGS_PER_PAGE + 9, "y")])
    out = bound_to_markdown(original)(doc, pages=[1, 0])
    assert [c["text"].rsplit("\n\n", 1)[1] for c in out] == ["y", "x"]
    assert calls == []


def test_no_pages_argument_means_every_page():
    original, calls = _recording_original()
    doc = _Doc([_Page(1, "a"), _Page(MAX_DRAWINGS_PER_PAGE + 1, "b")])
    out = bound_to_markdown(original)(doc)
    assert len(out) == 2 and calls[0]["pages"] == [0]
    assert bounded_extract.MAX_DRAWINGS_PER_PAGE == MAX_DRAWINGS_PER_PAGE
