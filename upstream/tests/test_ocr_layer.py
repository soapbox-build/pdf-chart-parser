"""Tests for the OCRmyPDF text-layer step and text-layer-based calibration.

The text-layer step is an optional dependency: tests that need OCRmyPDF to be
installed are guarded with importorskip, while the detection, graceful-failure,
and calibration-helper tests run with or without it.
"""

from __future__ import annotations

import fitz
import pytest

from pdf_chart_parser.ocr_layer import (
    add_text_layer,
    doc_needs_ocr,
    page_is_image_only,
)


def test_digital_pdf_is_not_image_only(synthetic_bar_pdf):
    doc = fitz.open(str(synthetic_bar_pdf))
    try:
        assert not page_is_image_only(doc[0])
        assert not doc_needs_ocr(doc)
    finally:
        doc.close()


def test_scanned_pdf_detected_as_image_only(synthetic_bar_raster_pdf):
    doc = fitz.open(str(synthetic_bar_raster_pdf))
    try:
        assert page_is_image_only(doc[0])
        assert doc_needs_ocr(doc)
    finally:
        doc.close()


def test_add_text_layer_graceful_on_bad_input():
    """Invalid input never raises: returns original bytes, applied=False, note.

    Robust whether OCRmyPDF is installed (it fails to parse) or absent (import
    fails) — either way the caller gets the original bytes back unchanged.
    """
    bad = b"not a pdf at all"
    out, applied, note = add_text_layer(bad)
    assert applied is False
    assert out == bad
    assert note is not None


def test_add_text_layer_applies_when_available(synthetic_bar_raster_pdf):
    """With OCRmyPDF and its binaries available, a scanned PDF gains a text layer.

    Skips when OCRmyPDF cannot actually run — the Python package may be
    importable while its system tools (tesseract, ghostscript) are absent.
    """
    pytest.importorskip(
        "ocrmypdf",
        reason="ocrmypdf not installed; install pdf-chart-parser[ocr] to enable",
    )
    pdf_bytes = synthetic_bar_raster_pdf.read_bytes()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    assert page_is_image_only(doc[0])  # precondition: no text layer yet
    doc.close()

    out, applied, note = add_text_layer(pdf_bytes)
    if not applied:
        pytest.skip(f"OCRmyPDF could not run ({note})")
    ocr_doc = fitz.open(stream=out, filetype="pdf")
    try:
        # The synthetic bar chart's tick labels / months should now be readable.
        assert not page_is_image_only(ocr_doc[0])
    finally:
        ocr_doc.close()


# ─── text-layer calibration helpers ─────────────────────────────────────────


def _page_with_axis_text() -> tuple[fitz.Document, fitz.Page]:
    """A 200x200 page with numeric ticks in the left gutter and bottom labels."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    # Left-gutter y-axis tick labels (x near 0).
    page.insert_text(fitz.Point(5, 50), "$100", fontsize=8)
    page.insert_text(fitz.Point(5, 100), "$50", fontsize=8)
    page.insert_text(fitz.Point(5, 150), "$0", fontsize=8)
    # A numeric label far from the left gutter must be ignored.
    page.insert_text(fitz.Point(150, 50), "999", fontsize=8)
    # Bottom x-axis labels, inserted out of left-to-right order.
    page.insert_text(fitz.Point(140, 190), "MAR", fontsize=8)
    page.insert_text(fitz.Point(20, 190), "JAN", fontsize=8)
    page.insert_text(fitz.Point(80, 190), "FEB", fontsize=8)
    return doc, page


def test_text_layer_y_axis_pairs_reads_left_gutter():
    from pdf_chart_parser.raster.cv_pipeline import _text_layer_y_axis_pairs

    doc, page = _page_with_axis_text()
    try:
        pairs = _text_layer_y_axis_pairs(page, zoom=1.0, w_img=200)
    finally:
        doc.close()

    values = sorted(v for v, _ in pairs)
    assert values == [0.0, 50.0, 100.0]
    assert 999.0 not in values  # right-side numeric excluded
    # y-position should increase as value decreases (axis runs bottom-up).
    by_value = {v: y for v, y in pairs}
    assert by_value[0.0] > by_value[100.0]


def test_text_layer_bottom_labels_ordered_left_to_right():
    from pdf_chart_parser.raster.cv_pipeline import _text_layer_bottom_labels

    doc, page = _page_with_axis_text()
    try:
        labels = _text_layer_bottom_labels(page, zoom=1.0, h_img=200)
    finally:
        doc.close()

    assert labels == ["JAN", "FEB", "MAR"]


# ─── document-reader integration ────────────────────────────────────────────


def test_document_reader_notes_ocr_on_scanned_pdf(synthetic_bar_raster_pdf):
    """A scanned PDF triggers the OCR step and records a note about it."""
    from pdf_chart_parser.document import extract_pdf_document

    result, _images, _native = extract_pdf_document(pdf_path=str(synthetic_bar_raster_pdf))
    assert any(
        "OCR" in note or "ocrmypdf" in note.lower() for note in result["notes"]
    )
