"""Keep extract_pdf_document inside the gateway's 120 s and off the event loop.

MEASURED 2026-10-04 (Adhoc-Tools-DocPath), Cloud Run logs for pdf-chart-parser and a local
run of the pinned upstream on the same documents:

1. The tools are plain `def` functions on the mcp 1.29 SDK's FastMCP, which calls a sync tool
   directly on the event loop. One slow extraction therefore froze the whole instance: every
   other request, initialize and tools/list included, queued behind it and ended as a 300 s
   Cloud Run 504 (a 15-minute window showed 14 consecutive 504s behind a handful of slow
   extractions). The caller, whose gateway gives up at 120 s, sees "session expired".
2. The slow extraction is pymupdf4llm.to_markdown on a vector-heavy page, a survey or drawing
   sheet with 15 000 to 20 000 vector drawings. Its cost grows much faster than the page's
   text: 3 000 drawings 0.2 s, 17 000 drawings 5 s, 20 600 drawings 7.6 s, and a 14 600-drawing
   page with 2 000 words 29 s on a laptop; the same call spent 215 531 ms on one 1-page
   document on the service's 2 CPUs. get_text("text") on the same page is 0.02 s.

So: (a) run every tool in a worker thread, so one slow call delays only itself; (b) send a
page with more than MAX_DRAWINGS_PER_PAGE vector drawings through PyMuPDF's plain text
extraction instead of pymupdf4llm's layout analysis, and say so in the page text, so a
caller sees the page was read without table/heading structure rather than not at all.
Pages under the limit take exactly the path they always did.
"""

from __future__ import annotations

import asyncio
import functools
import os

MAX_DRAWINGS_PER_PAGE = int(os.environ.get("PDF_MAX_DRAWINGS_PER_PAGE", "8000"))

PLAIN_TEXT_NOTE = (
    "[page {page}: {drawings} vector drawings; read as plain text, without table or heading "
    "structure, because layout analysis on a page this dense does not finish in time]"
)


def bound_to_markdown(original):
    """Wrap pymupdf4llm.to_markdown so a vector-dense page skips layout analysis."""

    @functools.wraps(original)
    def to_markdown(doc, *args, pages=None, **kwargs):
        indices = list(range(doc.page_count)) if pages is None else list(pages)
        heavy: dict[int, int] = {}
        for idx in indices:
            n = len(doc[idx].get_cdrawings())
            if n > MAX_DRAWINGS_PER_PAGE:
                heavy[idx] = n
        if not heavy:
            return original(doc, *args, pages=pages, **kwargs)
        light = [i for i in indices if i not in heavy]
        chunks = {}
        if light:
            for idx, chunk in zip(light, original(doc, *args, pages=light, **kwargs)):
                chunks[idx] = chunk
        for idx, n in heavy.items():
            text = doc[idx].get_text("text")
            chunks[idx] = {"text": PLAIN_TEXT_NOTE.format(page=idx + 1, drawings=n) + "\n\n" + text}
        return [chunks[i] for i in indices]

    return to_markdown


def run_tools_in_threads(mcp) -> int:
    """Make every registered sync tool run in a worker thread instead of on the event loop."""
    tools = mcp._tool_manager.list_tools()
    if not tools:
        raise RuntimeError("run_tools_in_threads: no tools registered; call it after registering them")
    count = 0
    for tool in tools:
        if tool.is_async:
            continue
        sync_fn = tool.fn

        async def threaded(*args, _fn=sync_fn, **kwargs):
            return await asyncio.to_thread(_fn, *args, **kwargs)

        tool.fn = threaded
        tool.is_async = True
        count += 1
    return count
