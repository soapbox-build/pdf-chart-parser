"""Add a searchable text layer to scanned PDFs via OCRmyPDF.

Scanned (image-only) PDFs carry no text layer, so the deterministic vector
calibration path and the Markdown document reader have nothing to read. This
module runs OCRmyPDF — which deskews/cleans each page and embeds an invisible
Tesseract text layer over the original image — turning a scanned PDF into one
the rest of the pipeline can treat like any born-digital document.

OCRmyPDF and its system tools (Ghostscript, qpdf, Tesseract) are an optional
dependency: every entry point degrades to a graceful no-op when the package or
its binaries are missing, returning the original bytes unchanged.
"""

from __future__ import annotations

import io
from typing import Any

# A page whose stripped text layer is shorter than this is treated as
# image-only (scanned). Mirrors document._IMAGE_ONLY_TEXT_THRESHOLD so both the
# chart pipeline and the document reader agree on what "needs OCR" means.
TEXT_LAYER_MIN_CHARS = 16


def page_is_image_only(page: Any) -> bool:
    """True when a page carries effectively no extractable text layer."""
    return len(page.get_text("text").strip()) < TEXT_LAYER_MIN_CHARS


def doc_needs_ocr(doc: Any, page_indices: list[int] | None = None) -> bool:
    """True when any of the considered pages is image-only (scanned).

    page_indices restricts the check to a subset (e.g. the pages a caller will
    actually return); None considers every page in the document.
    """
    indices = page_indices if page_indices is not None else range(doc.page_count)
    return any(page_is_image_only(doc[i]) for i in indices)


def add_text_layer(pdf_bytes: bytes) -> tuple[bytes, bool, str | None]:
    """Embed a searchable text layer into the scanned pages of a PDF.

    Runs OCRmyPDF with ``skip_text=True`` so pages that already have a text
    layer are passed through untouched and only image-only pages are OCR'd.

    Returns ``(pdf_bytes, applied, note)``. On success ``applied`` is True and
    ``pdf_bytes`` is the OCR'd document. If OCRmyPDF (or one of its binaries) is
    unavailable, or OCR fails for any reason, the original bytes are returned
    with ``applied=False`` and a human-readable note explaining why.
    """
    try:
        import ocrmypdf
    except ImportError:
        return (
            pdf_bytes,
            False,
            "scanned page(s) detected but OCRmyPDF is not installed; install "
            "pdf-chart-parser[ocr] for a searchable text layer",
        )

    inp = io.BytesIO(pdf_bytes)
    out = io.BytesIO()
    try:
        ocrmypdf.ocr(
            inp,
            out,
            # Only OCR pages that lack a text layer; leave digital pages as-is.
            skip_text=True,
            # Straighten skewed scans before OCR to lift recognition accuracy.
            deskew=True,
            # Keep the original page content (no PDF/A conversion) so vector
            # geometry the chart pipeline relies on is preserved verbatim.
            output_type="pdf",
            optimize=0,
            progress_bar=False,
        )
    except Exception as exc:
        return pdf_bytes, False, f"OCR text-layer step failed; using original PDF ({exc})"

    return out.getvalue(), True, "added a searchable text layer to scanned page(s) via OCRmyPDF"
