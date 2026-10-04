"""Generic multi-page PDF text + image extraction.

Generalizes the single-chart extraction in this server into a reusable document
reader: per-page Markdown text (via pymupdf4llm) plus optional rendered page
PNGs (via pymupdf). Any skill that needs to read a document can call this and 
get a deterministic, model-agnostic representation that works for vision and 
non-vision models alike.
"""

from __future__ import annotations

import base64
import re
import time

import fitz  # pymupdf
import pymupdf4llm
from mcp.types import BlobResourceContents, EmbeddedResource, ImageContent
from pydantic import BaseModel, Field

from pdf_chart_parser.io_utils import load_pdf_bytes
from pdf_chart_parser.logging_utils import elapsed_ms, get_logger, new_request_id
from pdf_chart_parser.ocr_layer import add_text_layer, doc_needs_ocr

_logger = get_logger(__name__)

# ─── Guardrails ───────────────────────────────────────────────────────────────
# Caps keep a single call bounded in time, memory, and response size so a large
# document can never hang the agent turn or blow the MCP response budget.
MAX_PAGES_PROCESSED = 60
# 50-page headroom buffer: real plan sets run up to ~40 pages in practice, and
# this cap must stay comfortably above that so images aren't silently dropped
# on a document only slightly larger than the ones already seen in practice.
# MAX_PAGES_PROCESSED (60) already covers this buffer with room to spare.
MAX_IMAGES_RENDERED = 50
MAX_TOTAL_IMAGE_BYTES = 18 * 1024 * 1024  # ~18 MB of base64-decoded PNG
MAX_IMAGE_DPI = 200
MIN_IMAGE_DPI = 36
# A page whose extracted text is shorter than this is treated as image-only
# (scanned) and gets a rendered PNG even when render_page_images is False, so a
# vision model can still read it.
_IMAGE_ONLY_TEXT_THRESHOLD = 16

# Native-PDF-attachment failsafe (attach_native_document / attach_native_pages
# below): a last-resort raw-PDF (or PDF page-range) MCP EmbeddedResource part,
# for when page-image rendering isn't enough — or isn't possible — and a
# caller needs the model to read the plan set's own vector content directly.
# This parser applies NO model-facing/provider-specific byte ceiling here —
# that number is provider-specific (Anthropic/OpenAI/Google all differ) and
# belongs in agent-service, measured against real provider limits, exactly
# how capMcpImages (not this parser) already owns the page-image byte/count
# cap. This is only a generous, provider-agnostic sanity cap against clearly
# pathological input (e.g. a caller mistakenly pointing this at a multi-GB
# file) — it is not a real ceiling and should never be tuned to match a
# provider's actual limit.
MAX_NATIVE_ATTACHMENT_BYTES = 200 * 1024 * 1024  # ~200 MB

# pymupdf4llm emits a placeholder like "**==> picture [252 x 252] intentionally
# omitted <==**" for pages whose only content is a raster image. That marker is
# itself the strongest signal a page is scanned/image-only, so strip it before
# measuring text length — otherwise the placeholder's own length masks the page
# as if it had real text and the scanned-page PNG fallback never fires.
_PICTURE_OMITTED_RE = re.compile(
    r"\*{0,2}==>\s*(?:picture|image)\b.*?omitted\s*<==\*{0,2}", re.IGNORECASE | re.DOTALL
)


def _meaningful_len(text: str) -> int:
    """Length of page text ignoring pymupdf4llm's image-omitted placeholders."""
    return len(_PICTURE_OMITTED_RE.sub("", text).strip())


class PdfPage(BaseModel):
    page: int = Field(description="1-based page number.")
    text: str = Field(description="Page content as Markdown (pymupdf4llm).")
    image_png_base64: str | None = Field(
        default=None,
        description=(
            "Deprecated, always null. Rendered page images are no longer inlined "
            "here as a duplicated base64 string — they are returned as separate "
            "MCP ImageContent parts alongside this document (see "
            "extract_pdf_document's return shape), each correlated to its page "
            "via that part's _meta.page (1-based). Kept only so a caller that "
            "still reads this field by name sees null instead of a missing key."
        ),
    )


class PdfDocument(BaseModel):
    total_pages: int = Field(description="Total page count of the source PDF.")
    pages: list[PdfPage]
    truncated: bool = Field(
        default=False,
        description="True if processing/image caps dropped pages or images from the result.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Human-readable notes about any capping or scanned-page handling.",
    )
    extraction_failed: bool = Field(
        default=False,
        description=(
            "True only when the PDF's bytes were fetched successfully but the "
            "parse stage itself then failed (a corrupted/encrypted PDF, or a "
            "pymupdf/pymupdf4llm crash) — pages/text above are empty in that "
            "case, and `reason` explains what failed. A fetch failure (bad "
            "URL, timeout, 404 — bytes never obtained) never reaches this far: "
            "it still raises, exactly as before this field existed, because "
            "there is nothing usable to report or attach."
        ),
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Present only when extraction_failed is true: a short description "
            "of the parse-stage exception, for a caller deciding whether the "
            "native-attachment failsafe result (if requested) is worth using "
            "in place of the normal per-page text/images."
        ),
    )


def _normalize_pages(pages: list[int] | None, total: int) -> tuple[list[int], bool]:
    """Map a 1-based page selection to 0-based indices within [0, total).

    Returns (zero_based_indices, truncated). Out-of-range pages are dropped.
    Default (None) selects all pages, capped at MAX_PAGES_PROCESSED.
    """
    if pages is None:
        selected = list(range(total))
    else:
        seen: set[int] = set()
        selected = []
        for p in pages:
            idx = p - 1
            if 0 <= idx < total and idx not in seen:
                seen.add(idx)
                selected.append(idx)
    truncated = False
    if len(selected) > MAX_PAGES_PROCESSED:
        selected = selected[:MAX_PAGES_PROCESSED]
        truncated = True
    return selected, truncated


def _build_native_attachments(
    data: bytes,
    attach_native_document: bool,
    attach_native_pages: list[int] | None,
    request_id: str,
) -> tuple[list[EmbeddedResource], list[str]]:
    """Build the native-PDF-attachment failsafe content part(s), if requested.

    `data` is the already-fetched raw PDF bytes — this never re-fetches
    anything. Whole-document wins if both `attach_native_document` and
    `attach_native_pages` are set (rather than erroring on the ambiguity),
    since a caller asking for "everything" is a reasonable superset of asking
    for a subset. Never raises: any failure while building the attachment
    (e.g. the source can't be re-opened by fitz to slice a page range because
    it's exactly the kind of malformed PDF this failsafe is likely to be
    invoked for) degrades to no attachment plus an explanatory note, so a
    native-attachment problem can never mask or replace whatever text/image
    result the caller already has from the rest of extract_pdf_document.

    Returns (attachments, notes) — attachments is empty when neither param is
    set, when nothing could be attached, or when the sanity cap trips.
    """
    if not attach_native_document and not attach_native_pages:
        return [], []

    notes: list[str] = []
    try:
        pages_meta: list[int] | str
        if attach_native_document:
            blob = data
            pages_meta = "all"
            mode = "whole-document"
        else:
            assert attach_native_pages is not None
            src = fitz.open(stream=data, filetype="pdf")
            try:
                zero_based, pages_truncated = _normalize_pages(
                    attach_native_pages, src.page_count
                )
                if pages_truncated:
                    notes.append(
                        f"native page-range attachment capped at "
                        f"{MAX_PAGES_PROCESSED} pages"
                    )
                if not zero_based:
                    notes.append(
                        "native page-range attachment skipped: none of the "
                        "requested attach_native_pages fell within the document"
                    )
                    return [], notes
                sliced = fitz.open()
                try:
                    for idx in zero_based:
                        sliced.insert_pdf(src, from_page=idx, to_page=idx)
                    blob = sliced.tobytes()
                finally:
                    sliced.close()
            finally:
                src.close()
            pages_meta = [idx + 1 for idx in zero_based]
            mode = "page-range"

        if len(blob) > MAX_NATIVE_ATTACHMENT_BYTES:
            notes.append(
                f"native attachment skipped: {len(blob)} bytes exceeds the "
                f"{MAX_NATIVE_ATTACHMENT_BYTES}-byte pathological-input sanity "
                "cap for a single in-memory PDF"
            )
            return [], notes

        meta = {"native_attachment": True, "pages": pages_meta}
        resource = BlobResourceContents(
            uri="attachment://planset-native.pdf",
            mimeType="application/pdf",
            blob=base64.b64encode(blob).decode("ascii"),
            # Constructed via the `_meta` wire alias, not the `meta` Python
            # field name — same footgun as ImageContent (document.py's
            # rendered-page-image path above): both ResourceContents and
            # EmbeddedResource declare `meta: ... = Field(alias="_meta")` with
            # `extra="allow"` and no `populate_by_name`, so a `meta=` kwarg
            # silently creates a spurious *extra* field and leaves the real
            # (alias-backed) `.meta` attribute None instead of raising.
            # Verified with a real round-trip, not assumed — see
            # tests/test_document.py's native-attachment meta-spelling test.
            _meta=meta,
        )
        embedded = EmbeddedResource(
            type="resource",
            resource=resource,
            _meta=meta,
        )
        _logger.info(
            "native PDF attachment produced",
            extra={
                "fields": {
                    "request_id": request_id,
                    "mode": mode,
                    "byte_size": len(blob),
                    "pages": pages_meta,
                }
            },
        )
        return [embedded], notes
    except Exception as exc:
        notes.append(f"native attachment failed: {type(exc).__name__}: {exc}")
        _logger.warning(
            "native PDF attachment failed",
            extra={
                "fields": {
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                }
            },
        )
        return [], notes


def extract_pdf_document(
    pdf_path: str | None = None,
    pdf_base64: str | None = None,
    pdf_url: str | None = None,
    pages: list[int] | None = None,
    render_page_images: bool = False,
    image_dpi: int = 100,
    attach_native_document: bool = False,
    attach_native_pages: list[int] | None = None,
) -> tuple[dict, list[ImageContent], list[EmbeddedResource]]:
    """Extract page text + optional page images from a PDF. Pure/deterministic.

    image_dpi defaults to 100: model image-token cost plateaus at roughly
    2,500 tokens/image above ~100 DPI (the vision patch cap), so a higher
    default buys the model nothing while costing bytes and latency, and makes
    it more likely a multi-page request trips MAX_TOTAL_IMAGE_BYTES before all
    pages are rendered.

    attach_native_document / attach_native_pages: a last-resort failsafe for
    when rendered page images aren't enough (or aren't possible) and a caller
    needs the model to read the native PDF's own vector content — see
    MAX_NATIVE_ATTACHMENT_BYTES above. `attach_native_pages` (1-based, same
    convention as `pages`) slices just those pages into a new in-memory PDF;
    `attach_native_document` attaches the whole original PDF's bytes verbatim.
    If both are set, whole-document wins. Neither is enabled by default —
    this is an opt-in failsafe, not part of the normal extraction path, and
    it applies no model-facing/provider-specific byte ceiling of its own
    (only a generous pathological-input sanity cap) — a real ceiling is the
    caller's job, exactly as capMcpImages already owns the page-image cap.

    Returns (document_dict, images, native_attachments): document_dict is the
    JSON-able PdfDocument shape (page text, never page image bytes); images is
    the list of rendered page images as real MCP ImageContent parts, in page
    order, each carrying its 1-based page number in `_meta={"page": ...}` so a
    caller can correlate an image back to its page's text block;
    native_attachments is zero or one MCP EmbeddedResource part (the
    attach_native_* failsafe output), present only when requested and
    buildable. The caller (this module's MCP tool wrapper in server.py) is
    responsible for assembling these into the final MCP content list — this
    function stays a plain, unit-testable Python function and only touches
    `mcp.types` for the image/resource-part construction itself.

    Bytes obtained vs. bytes usable: an exception fetching the PDF (bad URL,
    timeout, 404 — `load_pdf_bytes` itself failing) still raises, exactly as
    before — there are genuinely no bytes to report or attach. An exception
    in the *parse* stage after the bytes were already fetched (a corrupted/
    encrypted PDF, or a pymupdf/pymupdf4llm crash) does NOT raise: it returns
    a degraded `document_dict` with `extraction_failed=True` and a `reason`,
    plus whatever `attach_native_*` output could still be built from the
    bytes already in hand — a parse failure must never silently discard bytes
    that were actually obtained.
    """
    request_id = new_request_id()
    call_start = time.perf_counter()
    load_ms = 0.0

    # --- Fetch stage: bytes-in-hand, or a genuine hard failure. ---
    # This is the ONLY stage whose exception still propagates — once bytes
    # are obtained, every failure from here on is handled without raising
    # (see the parse-stage try/except below), because there is then always
    # something to report or attach.
    try:
        load_start = time.perf_counter()
        data = load_pdf_bytes(pdf_path=pdf_path, pdf_base64=pdf_base64, pdf_url=pdf_url)
        load_ms = elapsed_ms(load_start)
    except Exception as exc:
        _logger.warning(
            "extract_pdf_document fetch failed",
            extra={
                "fields": {
                    "request_id": request_id,
                    "total_ms": elapsed_ms(call_start),
                    "error_type": type(exc).__name__,
                }
            },
        )
        raise

    # --- Parse stage: bytes are in hand from here on. ---
    doc: fitz.Document | None = None
    try:
        dpi = max(MIN_IMAGE_DPI, min(MAX_IMAGE_DPI, image_dpi))

        notes: list[str] = []
        doc = fitz.open(stream=data, filetype="pdf")
        total = doc.page_count
        selected, truncated = _normalize_pages(pages, total)
        if truncated:
            notes.append(
                f"page selection capped at {MAX_PAGES_PROCESSED} pages (document has {total})"
            )

        # If any selected page is scanned/image-only, embed a searchable text
        # layer so its content comes back as real Markdown instead of nothing.
        if doc_needs_ocr(doc, selected):
            ocr_bytes, applied, ocr_note = add_text_layer(data)
            if ocr_note:
                notes.append(ocr_note)
            if applied:
                doc.close()
                doc = fitz.open(stream=ocr_bytes, filetype="pdf")

        # pymupdf4llm wants 0-based page numbers; request exactly the selected set.
        # use_ocr=False: pymupdf4llm's own page-analysis heuristic (image
        # variance / edge energy / vector-glyph-count) will otherwise decide a
        # page "needs OCR" and silently discard its real, correctly-extracted
        # native text layer in favor of a Tesseract read of a flattened raster
        # — a false positive on any page with an embedded photo, vicinity map,
        # or manufacturer screenshot, which is common on plan-set cover and
        # cutsheet pages. This is complementary to, not a regression of, the
        # doc_needs_ocr/add_text_layer step above: a genuinely scanned page has
        # already had a real OCR text layer burned in by that step by the time
        # to_markdown runs, so use_ocr=False here just stops pymupdf4llm from
        # redundantly (and sometimes incorrectly) re-deciding to OCR on its own.
        to_markdown_start = time.perf_counter()
        md_chunks = pymupdf4llm.to_markdown(doc, pages=selected, page_chunks=True, use_ocr=False)
        to_markdown_ms = elapsed_ms(to_markdown_start)
        # to_markdown returns one chunk per requested page, in request order.
        text_by_index = {
            idx: (chunk.get("text") or "") for idx, chunk in zip(selected, md_chunks)
        }

        rendered_images = 0
        total_image_bytes = 0
        total_text_chars = 0
        rasterize_ms = 0.0
        encode_ms = 0.0
        # Sticky flag: once the total-bytes cap trips on some page, stop even
        # attempting to rasterize later pages — rendering a PNG just to throw
        # it away (because the cap check used to happen *after* rendering) was
        # pure wasted work. Per-page text extraction above is unaffected: it
        # already ran for every page in `selected` before this loop starts.
        image_byte_cap_tripped = False
        out_pages: list[PdfPage] = []
        images: list[ImageContent] = []
        for idx in selected:
            text = text_by_index.get(idx, "")
            total_text_chars += len(text)

            is_image_only = _meaningful_len(text) < _IMAGE_ONLY_TEXT_THRESHOLD
            want_image = render_page_images or is_image_only

            if want_image:
                if rendered_images >= MAX_IMAGES_RENDERED:
                    truncated = True
                    notes.append(
                        f"image rendering capped at {MAX_IMAGES_RENDERED} images"
                    )
                elif image_byte_cap_tripped:
                    # Already know any further page would blow the byte cap —
                    # don't rasterize it just to discard the result.
                    truncated = True
                else:
                    raster_start = time.perf_counter()
                    png = doc.load_page(idx).get_pixmap(dpi=dpi).tobytes("png")
                    rasterize_ms += elapsed_ms(raster_start)
                    if total_image_bytes + len(png) > MAX_TOTAL_IMAGE_BYTES:
                        truncated = True
                        image_byte_cap_tripped = True
                        notes.append("image rendering stopped at the total-bytes cap")
                    else:
                        total_image_bytes += len(png)
                        rendered_images += 1
                        encode_start = time.perf_counter()
                        image_b64 = base64.b64encode(png).decode("ascii")
                        encode_ms += elapsed_ms(encode_start)
                        # Constructed via the `_meta` wire alias, not the `meta`
                        # Python field name: ImageContent has `extra="allow"`
                        # with no `populate_by_name`, so passing `meta=` as a
                        # kwarg silently creates a spurious *extra* field and
                        # leaves the real (alias-backed) `.meta` attribute
                        # `None` instead of raising — verified with a real
                        # round-trip, not assumed. `_meta` is the only kwarg
                        # spelling that actually populates `.meta`.
                        images.append(
                            ImageContent(
                                type="image",
                                data=image_b64,
                                mimeType="image/png",
                                _meta={"page": idx + 1},
                            )
                        )
                        if is_image_only and not render_page_images:
                            notes.append(
                                f"page {idx + 1} appears scanned/image-only — returning a rendered PNG"
                            )

            out_pages.append(PdfPage(page=idx + 1, text=text))

        if images:
            notes.append(
                "image_png_base64 is deprecated and always null; rendered page "
                "images are returned as separate MCP image content parts, each "
                "correlated to its page via that part's _meta.page"
            )

        native_attachments, native_notes = _build_native_attachments(
            data=data,
            attach_native_document=attach_native_document,
            attach_native_pages=attach_native_pages,
            request_id=request_id,
        )
        notes.extend(native_notes)

        result = PdfDocument(
            total_pages=total,
            pages=out_pages,
            truncated=truncated,
            notes=notes,
        ).model_dump()

        # Structured summary line: counts and timings only — never the
        # pdf_url (may be a presigned URL over customer PII) or any page
        # text/image bytes.
        _logger.info(
            "extract_pdf_document complete",
            extra={
                "fields": {
                    "request_id": request_id,
                    "total_pages": total,
                    "pages_processed": len(selected),
                    "images_rendered": rendered_images,
                    "truncated": truncated,
                    "total_text_chars": total_text_chars,
                    "total_image_bytes": total_image_bytes,
                    "load_ms": load_ms,
                    "to_markdown_ms": to_markdown_ms,
                    "rasterize_ms": round(rasterize_ms, 1),
                    "encode_ms": round(encode_ms, 1),
                    "total_ms": elapsed_ms(call_start),
                }
            },
        )
        return result, images, native_attachments
    except Exception as exc:
        # Bytes were already obtained (the fetch stage above already returned
        # successfully) — a parse-stage failure must not discard them. Report
        # the failure explicitly via extraction_failed/reason instead of
        # raising, and still attempt the requested native-attachment failsafe
        # from the bytes already in hand.
        _logger.warning(
            "extract_pdf_document parse failed",
            extra={
                "fields": {
                    "request_id": request_id,
                    "load_ms": load_ms,
                    "total_ms": elapsed_ms(call_start),
                    "error_type": type(exc).__name__,
                }
            },
        )
        native_attachments, native_notes = _build_native_attachments(
            data=data,
            attach_native_document=attach_native_document,
            attach_native_pages=attach_native_pages,
            request_id=request_id,
        )
        degraded = PdfDocument(
            total_pages=0,
            pages=[],
            truncated=False,
            notes=native_notes,
            extraction_failed=True,
            reason=f"{type(exc).__name__}: {exc}",
        ).model_dump()
        return degraded, [], native_attachments
    finally:
        if doc is not None:
            doc.close()
