"""bounded_extract.py: extractions run one at a time per instance, and a call that renders page
images reads a bounded number of pages at a bounded DPI.

Measured 2026-10-10 on the service's Cloud Run logs: 19 kills of "Memory limit of 2048 MiB
exceeded" in two windows, each dropping every request in flight on the instance. Concurrency was
160 per 2 GiB instance and every extraction ran in its own unbounded thread, so page-image
renders of a 45-page drawing set stacked until the instance was killed; that read failed 17 of 18
times. The upstream extractor is stubbed: these tests pin what the wrapper passes it and what it
tells the caller.
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bounded_extract import (  # noqa: E402
    MAX_RENDER_DPI,
    MAX_RENDER_PAGES,
    bound_page_images,
    run_tools_in_threads,
)

HOLD_S = 0.3


def _two_tool_server(state):
    mcp = FastMCP("concurrency-stub")

    def body():
        with state["lock"]:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(HOLD_S)
        with state["lock"]:
            state["now"] -= 1
        return "done"

    @mcp.tool()
    def extract_pdf_document(x: int = 0) -> str:
        return body()

    @mcp.tool()
    def count_pdf_page_chars(x: int = 0) -> str:
        return body()

    return mcp


def _fresh_state():
    return {"lock": threading.Lock(), "now": 0, "peak": 0}


async def _call_concurrently(mcp, name, n):
    await asyncio.gather(*(mcp.call_tool(name, {}) for _ in range(n)))


def test_extractions_on_one_instance_run_one_at_a_time():
    state = _fresh_state()
    mcp = _two_tool_server(state)
    run_tools_in_threads(mcp, slots=threading.BoundedSemaphore(1))
    asyncio.run(_call_concurrently(mcp, "extract_pdf_document", 3))
    assert state["peak"] == 1


def test_the_page_count_tool_is_not_queued_behind_extractions():
    # It returns counts only and renders nothing, so it keeps running in parallel.
    state = _fresh_state()
    mcp = _two_tool_server(state)
    run_tools_in_threads(mcp, slots=threading.BoundedSemaphore(1))
    asyncio.run(_call_concurrently(mcp, "count_pdf_page_chars", 3))
    assert state["peak"] >= 2


def test_queued_extractions_do_not_freeze_the_event_loop():
    state = _fresh_state()
    mcp = _two_tool_server(state)
    run_tools_in_threads(mcp, slots=threading.BoundedSemaphore(1))
    ticks = 0

    async def main():
        nonlocal ticks
        stop = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not stop.is_set():
                await asyncio.sleep(0.05)
                ticks += 1

        task = asyncio.create_task(ticker())
        await _call_concurrently(mcp, "extract_pdf_document", 2)
        stop.set()
        await task

    asyncio.run(main())
    assert ticks >= 5


def _recording_extract(total_pages):
    calls = []

    def original(**kwargs):
        calls.append(kwargs)
        pages = kwargs.get("pages")
        selected = list(range(1, total_pages + 1)) if pages is None else [p for p in pages if 1 <= p <= total_pages]
        doc = {
            "total_pages": total_pages,
            "pages": [{"page": p, "text": f"p{p}"} for p in selected],
            "truncated": False,
            "notes": [],
        }
        return doc, [f"img{p}" for p in selected], []

    return original, calls


def test_a_rendering_call_on_the_whole_document_reads_the_first_pages_and_names_the_rest():
    original, calls = _recording_extract(45)
    doc, images, native = bound_page_images(original)(pdf_url="u", render_page_images=True)
    assert calls[0]["pages"] == list(range(1, MAX_RENDER_PAGES + 1))
    assert [p["page"] for p in doc["pages"]] == list(range(1, MAX_RENDER_PAGES + 1))
    assert len(images) == MAX_RENDER_PAGES
    assert doc["truncated"] is True
    note = doc["notes"][-1]
    assert f"pages {MAX_RENDER_PAGES + 1}-45 were NOT read" in note
    assert "pages=[" in note


def test_a_rendering_call_with_too_many_pages_reads_them_in_the_order_given():
    original, calls = _recording_extract(45)
    asked = [30, 2, 2, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    doc, _, _ = bound_page_images(original)(pdf_url="u", pages=asked, render_page_images=True)
    assert calls[0]["pages"] == [30, 2, 31, 32, 33, 34, 35, 36, 37, 38]
    assert "pages 39-41 were NOT read" in doc["notes"][-1]


def test_pages_past_the_end_of_the_document_are_not_named_as_unread():
    original, _ = _recording_extract(12)
    doc, _, _ = bound_page_images(original)(pdf_url="u", render_page_images=True)
    assert f"pages {MAX_RENDER_PAGES + 1}-12 were NOT read" in doc["notes"][-1]
    original, _ = _recording_extract(5)
    doc, _, _ = bound_page_images(original)(pdf_url="u", pages=list(range(1, 15)), render_page_images=True)
    assert doc["notes"] == [] and doc["truncated"] is False


def test_a_rendering_call_within_the_cap_is_passed_through_untouched():
    original, calls = _recording_extract(45)
    asked = list(range(1, MAX_RENDER_PAGES + 1))
    doc, _, _ = bound_page_images(original)(pdf_url="u", pages=asked, render_page_images=True, image_dpi=MAX_RENDER_DPI)
    assert calls == [{"pdf_url": "u", "pages": asked, "render_page_images": True, "image_dpi": MAX_RENDER_DPI}]
    assert doc["notes"] == [] and doc["truncated"] is False


def test_a_text_only_call_is_not_capped():
    original, calls = _recording_extract(45)
    doc, _, _ = bound_page_images(original)(pdf_url="u")
    assert calls == [{"pdf_url": "u"}]
    assert len(doc["pages"]) == 45 and doc["notes"] == []


def test_a_dpi_above_the_cap_is_lowered_and_the_caller_is_told():
    original, calls = _recording_extract(3)
    doc, _, _ = bound_page_images(original)(pdf_url="u", render_page_images=True, image_dpi=300)
    assert calls[0]["image_dpi"] == MAX_RENDER_DPI
    assert any(f"rendered at {MAX_RENDER_DPI} dpi, not the 300 requested" in n for n in doc["notes"])


def test_the_dpi_cap_also_applies_to_the_scanned_page_images_of_a_text_call():
    # Upstream renders a scanned page whether or not render_page_images is set.
    original, calls = _recording_extract(3)
    bound_page_images(original)(pdf_url="u", image_dpi=300)
    assert calls[0]["image_dpi"] == MAX_RENDER_DPI


def test_a_schedule_can_be_read_at_200_dpi():
    # Task 1306: at 100 dpi a drawing set's equipment schedules were unreadable. Fails on main, whose cap was 100.
    assert MAX_RENDER_DPI == 200
    original, calls = _recording_extract(3)
    doc, _, _ = bound_page_images(original)(pdf_url="u", pages=[2], render_page_images=True, image_dpi=200)
    assert calls[0]["image_dpi"] == 200
    assert doc["notes"] == []


def test_a_failed_extraction_keeps_its_shape():
    def original(**kwargs):
        return {"total_pages": 0, "pages": [], "truncated": False, "notes": [], "extraction_failed": True}, [], []

    doc, images, native = bound_page_images(original)(pdf_url="u", render_page_images=True)
    assert doc["extraction_failed"] is True and images == [] and native == []
