"""FastMCP server exposing the extract_usage_chart tool."""

from __future__ import annotations

import os
from typing import Literal

from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations

from pdf_chart_parser.document import extract_pdf_document as _extract_pdf_document
from pdf_chart_parser.pipeline import extract_usage_chart as _run_pipeline

mcp = FastMCP(
    "usage-chart-extractor",
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8000")),
    # Each tool call is independent — no session tracking or SSE streaming needed.
    # json_response=True also satisfies AI agent clients that only send Accept: application/json.
    json_response=True,
    stateless_http=True,
)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        # pdf_url fetches the PDF over the network on that call path (pdf_path/
        # pdf_base64 stay local) — network access is conditional per call, not
        # unconditional, but this hint declares capability, not a per-call
        # guarantee, so True is the honest value given the tool as a whole.
        openWorldHint=True,
    ),
)
def extract_usage_chart(
    pdf_path: str | None = None,
    pdf_base64: str | None = None,
    pdf_url: str | None = None,
    page: int | None = None,
    chart_type: Literal["auto", "bar", "line", "hybrid"] = "auto",
    value_unit: Literal["auto", "dollars", "kwh"] = "auto",
    return_annotated_image: bool = True,
    render_dpi: int = 200,
) -> list:
    """Extract energy-usage chart data from a utility-bill PDF or image.

    Returns the full page text as Markdown, structured chart data as JSON,
    and optionally an annotated PNG of the chart region.

    Provide exactly one of pdf_path, pdf_base64, or pdf_url. Each accepts
    either a PDF or a photo/scan of a bill (JPEG, PNG, GIF, BMP, TIFF, or
    WebP) — an image input is converted into a one-page PDF internally
    before extraction runs, so the rest of the behavior below is unchanged.
    HEIC/HEIF images are not supported.

    `page`: 1-based page number (like `extract_pdf_document`). Omit to
    auto-detect the usage-chart page — recommended. Only pass this to override
    a wrong auto-detect.

    The result includes a 'series' list.  When the chart contains multiple
    utility types (e.g. electricity and gas on the same chart), each type is
    returned as a separate Series entry with its own 'id' ('s0', 's1', …) and
    'color'.  The caller is responsible for determining which series corresponds
    to which utility — use the bar colors, the series order, and the page
    Markdown context to make that determination.

    IMPORTANT — check `values_calibrated` before using any numbers:
    Some bills draw a chart whose bars/lines are visible but whose y-axis has
    no usable numeric scale (fewer than two readable tick labels, or a
    low-confidence fit). In that case this tool cannot compute real values, so
    every affected point's 'value' is returned as null rather than a fabricated
    number (0 or otherwise) — do not treat null as zero usage.

    - `values_calibrated` (bool): true only when every point's 'value' is a
      real, calibrated number. If false, DO NOT use 'series' as consumption or
      cost data — the bars/lines were detected but their values could not be
      measured.
    - `calibration_status` (string): "calibrated", "uncalibrated_axis" (no
      usable y-axis ticks were found), "low_confidence" (the axis or CV fit
      was too poor to trust), or "no_chart" (no chart was found at all).

    When `values_calibrated` is false, fall back to reading the document's
    full text instead — call the generic text-extraction tool in this same
    server (`extract_pdf_document`) on the same PDF and look for a printed
    monthly/13-month usage table, or a per-period daily-average value, in the
    page text. Only ask the caller/user for the numbers if neither the chart
    nor the text yields them.
    """
    result = _run_pipeline(
        pdf_path=pdf_path,
        pdf_base64=pdf_base64,
        pdf_url=pdf_url,
        page=page,
        chart_type=chart_type,
        value_unit=value_unit,
        return_annotated_image=return_annotated_image,
        render_dpi=render_dpi,
    )

    annotated_png: bytes | None = result.pop("annotated_png", None)

    out: list = [result]
    if return_annotated_image and annotated_png:
        out.append(Image(data=annotated_png, format="png"))
    return out


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        # Same reasoning as extract_usage_chart above: pdf_url makes this an
        # open-world fetch on that call path, even though pdf_path/pdf_base64
        # avoid the network entirely.
        openWorldHint=True,
    ),
)
def extract_pdf_document(
    pdf_path: str | None = None,
    pdf_base64: str | None = None,
    pdf_url: str | None = None,
    pages: list[int] | None = None,
    render_page_images: bool = False,
    image_dpi: int = 100,
    attach_native_document: bool = False,
    attach_native_pages: list[int] | None = None,
) -> list:
    """Extract per-page text and (optionally) page images from any PDF or image document.

    Generic, model-agnostic document reader for plan sets, permit packets, spec
    sheets, contracts, or any multi-page PDF — not just utility-bill charts. Use
    this whenever you need the textual content of a document, or page images for
    visual review.

    Provide exactly one of pdf_path, pdf_base64, or pdf_url. Each accepts either
    a PDF or a single image of a document (JPEG, PNG, GIF, BMP, TIFF, or WebP);
    an image input is converted into a one-page PDF internally before extraction
    runs, so it comes back as a single-page result. HEIC/HEIF images are not
    supported. `pages` is a 1-based list selecting specific pages (default: all).
    Set render_page_images=true to also get a base64 PNG of each page (e.g. for
    cover sheets, site plans, or single-line diagrams that need visual
    inspection). Scanned/image-only pages always come back with a rendered PNG
    so vision models can still read them.

    `attach_native_document`/`attach_native_pages` are a LAST-RESORT failsafe,
    off by default: when rendered page images aren't enough (or can't be
    produced) and the caller needs the model to see the native PDF itself.
    `attach_native_pages` (1-based, same convention as `pages`) attaches just
    those pages as a small sub-PDF; `attach_native_document` attaches the
    whole original PDF. If both are set, whole-document wins. Use sparingly —
    this bypasses the normal page-image path and its caller is responsible
    for any provider-specific size ceiling.

    Returns a list. The first item is
    { total_pages, pages: [{ page, text (Markdown) }], truncated, notes,
    extraction_failed, reason } — page text only, never page image bytes.
    `extraction_failed`/`reason` are populated only when the PDF's bytes were
    fetched successfully but parsing them then failed (never for a fetch
    failure, which still raises). It is followed by zero or more image
    content parts, one per rendered page image, in page order; each image's
    `_meta.page` gives its 1-based page number so it can be matched back to
    the corresponding page's text. Finally, zero or one native-PDF-attachment
    resource part (an MCP EmbeddedResource) is appended when
    attach_native_document/attach_native_pages was requested and could be
    built — its `_meta` carries `{native_attachment: true, pages: [...] |
    "all"}`. Page/image counts and total bytes are capped so a large document
    can never hang the turn.
    """
    doc, images, native_attachments = _extract_pdf_document(
        pdf_path=pdf_path,
        pdf_base64=pdf_base64,
        pdf_url=pdf_url,
        pages=pages,
        render_page_images=render_page_images,
        image_dpi=image_dpi,
        attach_native_document=attach_native_document,
        attach_native_pages=attach_native_pages,
    )
    out: list = [doc]
    out.extend(images)
    out.extend(native_attachments)
    return out


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
