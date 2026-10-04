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
import threading

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


# ---- OCR of scanned pages --------------------------------------------------------------------
#
# Upstream's extract_pdf_document OCRs a scanned page through ocrmypdf, but the image shipped
# without ocrmypdf, so add_text_layer answered "OCRmyPDF is not installed" and a scan came back as
# a rendered PNG and no text (measured 2026-10-04: requirements.lock held no ocrmypdf). Installing
# it alone would have been unbounded: upstream OCRs EVERY image-only page of the file, selected or
# not, with no page or time limit, inside a call the gateway cuts at 120 s. So the wrapper below
# (a) OCRs only the pages the caller selected, at most MAX_OCR_PAGES of them, each under
# OCR_PAGE_TIMEOUT_S, and says what it did not OCR; and (b) marks every OCR'd page's text as OCR,
# because recognised text is not the document's own text layer and can be wrong.

MAX_OCR_PAGES = int(os.environ.get("PDF_MAX_OCR_PAGES", "12"))
OCR_PAGE_TIMEOUT_S = float(os.environ.get("PDF_OCR_PAGE_TIMEOUT_S", "30"))
IMAGE_ONLY_CHARS = 16  # upstream's own threshold for "no text layer"

OCR_NOTE = "[page {page}: text recognised by OCR from a scan; it may contain errors]"

_call = threading.local()  # selected pages and the pages OCR'd by THIS thread's current call


def _open_pdf(pdf_bytes: bytes):
    import pymupdf

    return pymupdf.open(stream=pdf_bytes, filetype="pdf")


def _run_ocr(pdf_bytes: bytes, pages_1based: list[int]) -> bytes:
    import io

    import ocrmypdf

    out = io.BytesIO()
    ocrmypdf.ocr(
        io.BytesIO(pdf_bytes),
        out,
        pages=",".join(str(p) for p in pages_1based),
        skip_text=True,
        deskew=False,
        output_type="pdf",
        optimize=0,
        progress_bar=False,
        tesseract_timeout=OCR_PAGE_TIMEOUT_S,
        jobs=2,
    )
    return out.getvalue()


def record_selected_pages(original):
    """Wrap document.doc_needs_ocr, which is called once per extraction with the selected pages."""

    @functools.wraps(original)
    def doc_needs_ocr(doc, page_indices=None):
        _call.selected = list(range(doc.page_count)) if page_indices is None else list(page_indices)
        _call.ocr_pages = set()
        return original(doc, page_indices)

    return doc_needs_ocr


def bound_add_text_layer(original):
    """Wrap document.add_text_layer: OCR the selected scanned pages only, at most MAX_OCR_PAGES."""

    @functools.wraps(original)
    def add_text_layer(pdf_bytes):
        selected = getattr(_call, "selected", None)
        if selected is None:  # called outside an extraction: keep upstream's behaviour
            return original(pdf_bytes)
        try:
            doc = _open_pdf(pdf_bytes)
            scanned = [i for i in selected if len(doc[i].get_text("text").strip()) < IMAGE_ONLY_CHARS]
        except Exception as exc:  # unreadable here means unreadable upstream too; say so, do not guess
            return pdf_bytes, False, f"OCR skipped: could not inspect the scanned pages ({exc})"
        if not scanned:
            return pdf_bytes, False, None
        todo, left = scanned[:MAX_OCR_PAGES], scanned[MAX_OCR_PAGES:]
        try:
            ocr_bytes = _run_ocr(pdf_bytes, [i + 1 for i in todo])
        except ImportError:
            return pdf_bytes, False, "scanned page(s) detected but OCRmyPDF is not installed; page images only"
        except Exception as exc:
            return pdf_bytes, False, f"OCR failed or timed out; scanned pages are returned as page images only ({exc})"
        _call.ocr_pages = set(todo)
        note = f"OCR applied to scanned page(s) {', '.join(str(i + 1) for i in todo)}; their text is recognised, not the document's own"
        if left:
            note += (
                f"; {len(left)} further scanned page(s) ({', '.join(str(i + 1) for i in left)}) were NOT OCR'd because a "
                f"call OCRs at most {MAX_OCR_PAGES}: pass pages=[...] to choose them, they come back as page images"
            )
        return ocr_bytes, True, note

    return add_text_layer


def disclose_ocr(to_markdown):
    """Wrap to_markdown so the text of an OCR'd page says it is OCR.

    Only when OCR produced real text: upstream renders a page image for a page whose text is under
    its image-only threshold, and a disclosure line must not be what stops that.
    """

    @functools.wraps(to_markdown)
    def wrapped(doc, *args, pages=None, **kwargs):
        chunks = to_markdown(doc, *args, pages=pages, **kwargs)
        ocr_pages = getattr(_call, "ocr_pages", set())
        if not ocr_pages:
            return chunks
        indices = list(range(doc.page_count)) if pages is None else list(pages)
        for idx, chunk in zip(indices, chunks):
            text = chunk.get("text") or ""
            if idx in ocr_pages and len(text.strip()) >= IMAGE_ONLY_CHARS:
                chunk["text"] = OCR_NOTE.format(page=idx + 1) + "\n\n" + text
        return chunks

    return wrapped


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
