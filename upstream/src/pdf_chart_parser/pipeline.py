"""Orchestrator: load PDF, detect chart type, run extraction, assemble result."""

from __future__ import annotations

from typing import Any, Literal

import fitz
import pymupdf4llm

from pdf_chart_parser.annotate import annotate_chart
from pdf_chart_parser.io_utils import load_pdf_bytes
from pdf_chart_parser.models import Axes, Series
from pdf_chart_parser.ocr_layer import add_text_layer, doc_needs_ocr
from pdf_chart_parser.vector.bars import extract_bars
from pdf_chart_parser.vector.calibrate import calibrate_axes
from pdf_chart_parser.vector.drawings import collect_drawings
from pdf_chart_parser.vector.lines import extract_lines
from pdf_chart_parser.vector.locate import horizontal_gridline_ys, locate_chart
from pdf_chart_parser.vector.text import collect_text_spans


def extract_usage_chart(
    pdf_path: str | None = None,
    pdf_base64: str | None = None,
    pdf_url: str | None = None,
    page: int | None = None,
    chart_type: Literal["auto", "bar", "line", "hybrid"] = "auto",
    value_unit: Literal["auto", "dollars", "kwh"] = "auto",
    return_annotated_image: bool = True,
    render_dpi: int = 200,
) -> dict[str, Any]:
    """End-to-end extraction: returns a dict matching the ExtractionResult schema.

    `page`, when given, is 1-based (matching the sibling `extract_pdf_document`
    tool) — omit it to auto-detect the chart page.

    Also includes an 'annotated_png' key (bytes | None) when return_annotated_image=True.
    """
    try:
        pdf_bytes = load_pdf_bytes(pdf_path=pdf_path, pdf_base64=pdf_base64, pdf_url=pdf_url)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        result = _failed_result([f"failed to load PDF: {exc}"])
        result["page_markdown"] = ""
        result["page"] = 0
        return result

    # Scanned PDFs carry no text layer, so neither the vector calibration path
    # nor page_markdown can read tick labels. Embed a searchable text layer via
    # OCR first so a scanned bill flows through the same path as a digital one.
    ocr_note: str | None = None
    if doc_needs_ocr(doc):
        ocr_bytes, applied, ocr_note = add_text_layer(pdf_bytes)
        if applied:
            doc.close()
            doc = fitz.open(stream=ocr_bytes, filetype="pdf")

    # The public `page` argument is 1-based; convert to a 0-based hint once,
    # here, so every internal helper deals in 0-based indices.
    page_hint = None if page is None else page - 1

    # Select the target page first, then render markdown scoped to that page so
    # page_markdown and the reported page index describe the same page.
    target_page = _select_page(doc, page_hint)
    page_markdown = _extract_page_markdown(doc, target_page)

    result = _try_vector(
        doc, target_page, chart_type, value_unit, render_dpi, return_annotated_image
    )
    if ocr_note:
        result["warnings"].insert(0, ocr_note)
    result["page_markdown"] = page_markdown
    result["page"] = target_page + 1  # report 1-based, matching the input contract

    doc.close()
    return result


def _extract_page_markdown(doc: Any, page_index: int) -> str:
    """Return LLM-friendly markdown for the selected page.

    Rendered on an isolated copy of `doc` (`doc.tobytes()` round-trip), never
    on the caller's own doc: `pymupdf4llm.to_markdown` can invoke OCR that
    rewrites a page's text layer, which would destroy the vector text spans
    the calibration path in `_try_vector` reads from the *same* `doc`. Working
    on a throwaway copy keeps that guarantee regardless of call order.

    OCR is also skipped outright (`use_ocr=False`) whenever the page already
    carries an extractable text layer — either natively, or because the
    `doc_needs_ocr`/`add_text_layer` step above already ran OCR up front for a
    genuinely scanned page. A born-digital page never needs to be re-rendered
    through Tesseract: pymupdf4llm's own page analysis can still flag such a
    page as "image-heavy" (e.g. a bill whose chart page also carries a photo
    or logo) and reach for OCR anyway, which is both unnecessary and slow.
    """
    page_has_text = bool(doc[page_index].get_text("text").strip())
    tmp = fitz.open(stream=doc.tobytes(), filetype="pdf")
    try:
        return pymupdf4llm.to_markdown(tmp, pages=[page_index], use_ocr=not page_has_text)
    except Exception:
        return ""
    finally:
        tmp.close()


def _select_page(doc: Any, page_hint: int | None) -> int:
    """Return the 0-based page index most likely to contain a usage chart.

    `page_hint`, when given, is already 0-based (the public `page` argument is
    converted once by the caller). A page that yields an actual detectable
    chart (bars or lines) is preferred over one that merely matches usage
    keywords, so bills whose bars are drawn as filled line-quads (not 're'
    rectangles) still resolve to the chart page. Keyword matches act as the
    tie-breaker / fallback when no page detects a chart.
    """
    if page_hint is not None:
        return max(0, min(page_hint, len(doc) - 1))

    best_page = 0
    best_score = -1.0
    for i in range(len(doc)):
        pg = doc[i]
        text = pg.get_text("text").lower()
        keyword_score = sum(
            kw in text
            for kw in ("kwh", "usage", "kw", "$", "electric", "gas", "billing", "charges")
        )

        chart_elements = 0
        try:
            drawings = collect_drawings(pg)
            spans = collect_text_spans(pg)
            location = locate_chart(drawings, spans, "auto")
            if location is not None:
                _, _, bar_rects, line_paths, plot_rect = location
                # Require the plot to span a meaningful width so small logo or
                # icon squiggles (which can pass the polyline filters) don't
                # masquerade as charts. Weight by bar count and distinct line
                # paths rather than raw vertices, so a noisy multi-vertex glyph
                # cannot outrank a genuine bar chart.
                if plot_rect.width >= 80:
                    chart_elements = len(bar_rects) + len(line_paths)
        except Exception:
            chart_elements = 0

        # A detected chart dominates the score; keywords only break ties or
        # rank pages where nothing chart-like was found.
        score = (100.0 if chart_elements else 0.0) + chart_elements + keyword_score * 0.1
        if score > best_score:
            best_score = score
            best_page = i

    return best_page


def _try_vector(
    doc: Any,
    page_index: int,
    chart_type_hint: str,
    value_unit_hint: str,
    render_dpi: int,
    return_annotated_image: bool,
) -> dict[str, Any]:
    """Attempt vector extraction; fall back to raster on failure."""
    warnings: list[str] = []
    page = doc[page_index]

    raw_text = page.get_text("text").strip()
    if not raw_text:
        warnings.append("page appears scanned or raster; text layer empty")
        return _try_raster(
            doc,
            page_index,
            chart_type_hint,
            value_unit_hint,
            render_dpi,
            return_annotated_image,
            warnings,
        )

    try:
        drawings = collect_drawings(page)
        spans = collect_text_spans(page)
        location = locate_chart(drawings, spans, chart_type_hint)

        if location is None:
            warnings.append("no qualifying chart region found via vector; falling back to raster")
            return _try_raster(
                doc,
                page_index,
                chart_type_hint,
                value_unit_hint,
                render_dpi,
                return_annotated_image,
                warnings,
            )

        chart_rect, detected_type, bar_rects, line_paths, plot_rect = location
        gridline_ys = horizontal_gridline_ys(drawings["paths"], plot_rect)
        axes, calibration_warnings = calibrate_axes(
            spans,
            chart_rect,
            value_unit_hint,
            plot_rect=plot_rect,
            bar_rects=bar_rects,
            gridline_ys=gridline_ys,
        )
        warnings.extend(calibration_warnings)

        series = []
        if bar_rects:
            bar_series, bar_warnings = extract_bars(bar_rects, axes, chart_rect, spans)
            warnings.extend(bar_warnings)
            series.extend(bar_series)

        if line_paths:
            line_series, line_warnings = extract_lines(
                line_paths, axes, spans, chart_rect, plot_rect=plot_rect
            )
            warnings.extend(line_warnings)
            series.extend(line_series)

        annotated_png = None
        if return_annotated_image:
            try:
                annotated_png = annotate_chart(
                    page, chart_rect, series, axes, render_dpi, detected_type
                )
            except Exception as exc:
                warnings.append(f"annotation failed: {exc}")

        confidence = _compute_confidence(axes, series, warnings)
        values_calibrated, calibration_status = _calibration_status(axes, series)

        return {
            "chart_found": True,
            "method": "vector",
            "chart_type": detected_type,
            "axes": axes.model_dump(exclude_none=True),
            "series": [s.model_dump() for s in series],
            "warnings": warnings,
            "confidence": confidence,
            "values_calibrated": values_calibrated,
            "calibration_status": calibration_status,
            "annotated_png": annotated_png,
        }

    except Exception as exc:
        warnings.append(f"vector extraction failed: {exc}")
        return _try_raster(
            doc,
            page_index,
            chart_type_hint,
            value_unit_hint,
            render_dpi,
            return_annotated_image,
            warnings,
        )


def _try_raster(
    doc: Any,
    page_index: int,
    chart_type_hint: str,
    value_unit_hint: str,
    render_dpi: int,
    return_annotated_image: bool,
    warnings: list[str],
) -> dict[str, Any]:
    """Attempt raster CV/OCR extraction. Returns failed result if unavailable."""
    # Import is deferred: cv_pipeline requires the optional [raster] extra (cv2).
    # Keeping this inside the function is intentional — it guards against ImportError
    # when only the default (vector-only) install is present.
    try:
        from pdf_chart_parser.raster.cv_pipeline import extract_raster
    except ImportError:
        warnings.append("raster extra not installed; install pdf-chart-parser[raster] for fallback")
        return _failed_result(warnings)

    try:
        return extract_raster(
            doc,
            page_index,
            chart_type_hint,
            value_unit_hint,
            render_dpi,
            return_annotated_image,
            warnings,
        )
    except Exception as exc:
        warnings.append(f"raster extraction failed: {exc}")
        return _failed_result(warnings)


def _failed_result(warnings: list[str]) -> dict[str, Any]:
    return {
        "chart_found": False,
        "method": "failed",
        "chart_type": None,
        "axes": {},
        "series": [],
        "warnings": warnings,
        "confidence": 0.0,
        "values_calibrated": False,
        "calibration_status": "no_chart",
        "annotated_png": None,
    }


def _calibration_status(axes: Axes, series: list[Series]) -> tuple[bool, str]:
    """Decide whether the series' values are real data or should be distrusted.

    Bars and lines both null out any point they cannot derive from either a
    well-fit y-axis scale or a printed value label, so a chart is only
    "calibrated" when every returned point has a real value.
    """
    points = [p for s in series for p in s.points]
    if points and all(p.value is not None for p in points):
        return True, "calibrated"
    if len(axes.y_primary.points) >= 2:
        return False, "low_confidence"
    return False, "uncalibrated_axis"


def _compute_confidence(axes: Axes, series: list, warnings: list[str]) -> float:
    """Heuristic overall confidence score."""
    if not series:
        return 0.0
    # Uncalibrated y-axis (< 2 tick labels) means every value defaults to 0.0;
    # cap confidence low so a bogus all-zero series is not reported as reliable.
    if len(getattr(axes.y_primary, "points", [])) < 2:
        return 0.2
    base = 1.0
    r2 = getattr(axes.y_primary, "r_squared", 0.0)
    if r2 < 0.999:
        base -= 0.1
    if any("secondary" in w for w in warnings):
        base -= 0.02
    series_confidences = [s.confidence for s in series]
    if series_confidences:
        avg_series = sum(series_confidences) / len(series_confidences)
        base = (base + avg_series) / 2
    return round(max(0.0, min(1.0, base)), 4)
