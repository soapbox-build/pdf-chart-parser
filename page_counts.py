"""count_pdf_page_chars: per-page character counts for a PDF, and NO text.

Soapbox's addition, applied by auth_app.py. The marketplace's document-coverage gate needs to know
which pages of a document have a usable text layer (lib-document-coverage/coverage_gate.py,
page_chars): a page under the floor is a scan or a drawing and has to be viewed, not read as text.
On 2026-10-06 a seat could not produce those counts for a large document: Dragon's
get_document_content returned the whole text, the harness spilled it to a file the seat may only
Read, and counting characters needs the exact text, which a Read transcription does not keep. So
this tool counts on the server and returns numbers only. A document's text never enters the
caller's transcript, which is also the reason it returns no text at all, not even a sample.

Counts are PyMuPDF's plain-text extraction (page.get_text("text"), stripped), with no OCR: the
question is whether the file's own text layer holds the page. They are a different extraction from
the platform's markdown, so they are compared only with the floor, never with total_chars. The
floor is the coverage gate's IMAGE_ONLY_CHARS_PER_PAGE (100), measured 2026-10-06 over 87 ready
documents: image-only documents at 0 to 13.2 characters a page, the thinnest text document at 229.
"""

from __future__ import annotations

from urllib.parse import urlparse

# Kept equal to coverage_gate.IMAGE_ONLY_CHARS_PER_PAGE (soapbox-marketplace); see the docstring.
IMAGE_ONLY_CHARS_PER_PAGE = 100
# The same cap upstream's own fetch applies to pdf_url.
MAX_PDF_BYTES = 50 * 1024 * 1024
FETCH_TIMEOUT_S = 60.0

TOOL_NAME = "count_pdf_page_chars"
TOOL_DESCRIPTION = (
    "Count the extracted text characters on every page of a PDF, and flag each page whose text "
    f"layer is under {IMAGE_ONLY_CHARS_PER_PAGE} characters as image_only (a scan, photograph or "
    "drawing that has to be viewed, not read as text). Returns numbers only, never the document's "
    "text. `page_chars` is shaped for the document-coverage ledger's page_chars field. pdf_url is "
    "an http(s) URL to the file, for example from get_document_download_url."
)


def page_counts(doc) -> dict:
    """Per-page counts for an open document: any iterable of pages answering get_text("text")."""
    pages = []
    for number, page in enumerate(doc, start=1):
        chars = len((page.get_text("text") or "").strip())
        pages.append({"page": number, "chars": chars, "image_only": chars < IMAGE_ONLY_CHARS_PER_PAGE})
    image_only = [p["page"] for p in pages if p["image_only"]]
    return {
        "page_count": len(pages),
        "pages": pages,
        "page_chars": {str(p["page"]): p["chars"] for p in pages},
        "image_only_pages": image_only,
        "image_only_chars_per_page_floor": IMAGE_ONLY_CHARS_PER_PAGE,
        "extractor": "pymupdf get_text('text'), stripped, no OCR",
        "note": (
            "Counts only; no text is returned. Compare each page with the floor, not with the "
            "platform's total_chars: they are different extractions of the same file."
        ),
    }


def fetch_pdf(pdf_url: str) -> bytes:
    """The file at an http(s) URL, at most MAX_PDF_BYTES."""
    import httpx

    scheme = urlparse(pdf_url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"pdf_url must be an http(s) URL, not {scheme or 'a bare path'}")
    body = bytearray()
    with httpx.stream("GET", pdf_url, follow_redirects=True, timeout=FETCH_TIMEOUT_S) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > MAX_PDF_BYTES:
                raise ValueError(f"the PDF is over {MAX_PDF_BYTES // (1024 * 1024)} MB")
    return bytes(body)


def open_pdf(data: bytes):
    import pymupdf

    return pymupdf.open(stream=data, filetype="pdf")


def register_page_counts(mcp, fetch=fetch_pdf, opener=open_pdf) -> None:
    """Add count_pdf_page_chars to `mcp`. Called before strict_tool_arguments and
    run_tools_in_threads, so it refuses undeclared arguments and runs off the event loop like the
    upstream tools."""

    def count_pdf_page_chars(pdf_url: str) -> dict:
        doc = opener(fetch(pdf_url))
        try:
            return page_counts(doc)
        finally:
            close = getattr(doc, "close", None)
            if close:
                close()

    mcp._tool_manager.add_tool(count_pdf_page_chars, name=TOOL_NAME, description=TOOL_DESCRIPTION)
