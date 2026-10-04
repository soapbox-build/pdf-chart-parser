"""Regression tests for two related pipeline defects:

1. `pymupdf4llm.to_markdown` can run OCR that rewrites a page's text layer as
   a side effect. If that call ever runs against the same `doc` object the
   vector chart calibrator reads from, the calibrator loses the numeric
   tick-label spans it depends on. `_extract_page_markdown` must render
   markdown on an isolated copy of the doc, and must not request OCR at all
   for a page that already has a native text layer.
2. `extract_usage_chart`'s `page` argument must be 1-based, matching the
   sibling `extract_pdf_document` tool, so a caller cannot get a different
   answer depending on which tool it asks.

The fixture below is a minimal born-digital PDF: a vector bar chart with
numeric y-axis tick labels, sharing a page with an embedded photo-like image.
pymupdf4llm's own page analysis classifies pages like this as "image-heavy"
and is tempted to reach for OCR — exactly the scenario that must never
mutate/render a native digital page.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
import numpy as np
import pymupdf4llm.helpers.document_layout as pymupdf4llm_document_layout
import pytest
from PIL import Image

from pdf_chart_parser import pipeline
from pdf_chart_parser.pipeline import extract_usage_chart
from pdf_chart_parser.vector.text import collect_text_spans

MONTHS = ["Mar", "Apr", "May", "Jun", "Jul"]
BAR_VALUES = [12.3, 34.5, 50.2, 61.7, 45.0]
Y_TICKS = [0, 23, 46, 68]

CHART_PAGE_INDEX = 1  # 0-based: the fixture's second page
CHART_PAGE_NUMBER = CHART_PAGE_INDEX + 1  # 1-based: what callers pass/see


def _make_digital_bill_pdf(path: Path) -> None:
    """Build the fixture described above: a chart-less cover page (so page
    selection — auto-detect vs. an explicit `page` — is meaningfully
    exercised) followed by a page with a native vector bar chart plus an
    embedded photo-like image.
    """
    doc = fitz.open()
    cover = doc.new_page(width=400, height=400)
    cover.insert_text(fitz.Point(50, 50), "Cover Page", fontsize=14)
    cover.insert_text(fitz.Point(50, 80), "Account summary and welcome message.", fontsize=10)

    page = doc.new_page(width=400, height=400)
    chart_left, chart_right = 80.0, 350.0
    chart_top, chart_bottom = 70.0, 320.0
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top
    max_val = max(Y_TICKS)

    for tick in Y_TICKS:
        y = chart_bottom - (tick / max_val) * chart_h
        page.draw_line(fitz.Point(chart_left - 5, y), fitz.Point(chart_left, y))
        page.insert_text(fitz.Point(chart_left - 30, y + 4), str(tick), fontsize=9)
        page.draw_line(
            fitz.Point(chart_left, y),
            fitz.Point(chart_right, y),
            color=(0.8, 0.8, 0.8),
            width=0.5,
        )

    page.draw_line(
        fitz.Point(chart_left, chart_top), fitz.Point(chart_left, chart_bottom), width=1.5
    )
    page.draw_line(
        fitz.Point(chart_left, chart_bottom), fitz.Point(chart_right, chart_bottom), width=1.5
    )

    n = len(BAR_VALUES)
    bar_spacing = chart_w / n
    bar_width = bar_spacing * 0.6
    for i, val in enumerate(BAR_VALUES):
        bar_top = chart_bottom - (val / max_val) * chart_h
        x0 = chart_left + i * bar_spacing + (bar_spacing - bar_width) / 2
        x1 = x0 + bar_width
        page.draw_rect(fitz.Rect(x0, bar_top, x1, chart_bottom), fill=(0.2, 0.4, 0.7), color=None)
        page.insert_text(fitz.Point(x0, chart_bottom + 12), MONTHS[i], fontsize=8)

    page.insert_text(fitz.Point(90, 55), "Daily average kWh usage", fontsize=11)

    # Photo-like embedded image: high pixel variance/edge energy is the same
    # signal pymupdf4llm's page analyzer uses to flag a page as image-heavy
    # and reach for OCR (a fixed seed keeps the fixture deterministic).
    rng = np.random.default_rng(42)
    arr = (rng.random((150, 180, 3)) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    image_rect = fitz.Rect(chart_left, chart_bottom + 30, chart_left + 180, chart_bottom + 180)
    page.insert_image(image_rect, stream=buf.getvalue())

    doc.save(str(path))
    doc.close()


@pytest.fixture
def digital_bill_pdf(tmp_path) -> Path:
    path = tmp_path / "digital_bill.pdf"
    _make_digital_bill_pdf(path)
    return path


def test_markdown_does_not_mutate_doc(digital_bill_pdf, monkeypatch):
    """Regression guard for root cause A: `_extract_page_markdown` must
    render markdown on an isolated copy of the doc, never the caller's own
    doc.

    Rather than rely on the exact (version- and tesseract-availability-
    sensitive) conditions under which the real `to_markdown` happens to
    mutate a page, simulate a maximally hostile implementation that wipes
    whatever doc it is handed, and confirm the caller's own doc survives
    untouched.
    """
    doc = fitz.open(str(digital_bill_pdf))
    before = len(collect_text_spans(doc[CHART_PAGE_INDEX]))
    assert before > 0

    def _hostile_to_markdown(mutated_doc, pages, **kwargs):
        page = mutated_doc[pages[0]]
        page.add_redact_annot(page.rect)
        page.apply_redactions()
        return "mutated"

    monkeypatch.setattr(pipeline.pymupdf4llm, "to_markdown", _hostile_to_markdown)

    md = pipeline._extract_page_markdown(doc, CHART_PAGE_INDEX)

    assert md == "mutated"  # the hostile stub really ran
    after = len(collect_text_spans(doc[CHART_PAGE_INDEX]))
    assert after == before  # ...but the caller's own doc was never touched
    doc.close()


def test_digital_bill_calibrates(digital_bill_pdf):
    """End-to-end guard: a native digital bill whose chart page also carries
    an image must still calibrate via the vector path. This is the reported
    bug's shape: the chart was detected but came back `values_calibrated:
    false` with every value null.
    """
    result = extract_usage_chart(pdf_path=str(digital_bill_pdf), return_annotated_image=False)

    assert result["method"] == "vector"
    assert result["values_calibrated"] is True
    assert result["calibration_status"] == "calibrated"
    assert result["page"] == CHART_PAGE_NUMBER

    series = result["series"][0]
    values = [p["value"] for p in series["points"]]
    assert len(values) == len(BAR_VALUES)
    assert all(v is not None for v in values)
    assert values == pytest.approx(BAR_VALUES, abs=1.0)


def test_page_argument_is_1_based(digital_bill_pdf):
    """Part B guard: `page` is 1-based, matching `extract_pdf_document`.

    The correct 1-based chart page must resolve to the same chart as
    auto-detect and echo back the same 1-based number. A deliberately wrong
    1-based page (the chart-less cover page) must not calibrate, proving the
    index conversion is not silently absorbing an off-by-one error.
    """
    auto = extract_usage_chart(pdf_path=str(digital_bill_pdf), return_annotated_image=False)
    explicit = extract_usage_chart(
        pdf_path=str(digital_bill_pdf), page=CHART_PAGE_NUMBER, return_annotated_image=False
    )
    assert explicit["page"] == CHART_PAGE_NUMBER
    assert explicit["values_calibrated"] is True
    assert explicit["series"] == auto["series"]

    wrong_page = extract_usage_chart(
        pdf_path=str(digital_bill_pdf),
        page=1,  # the chart-less cover page
        return_annotated_image=False,
    )
    assert wrong_page["page"] == 1
    assert wrong_page["values_calibrated"] is False


def test_no_ocr_on_digital_pdf(digital_bill_pdf, monkeypatch):
    """Part A2 guard: a page with a native text layer must never be handed
    to `to_markdown` with OCR enabled — a born-digital page should never be
    rendered/recognized, even one pymupdf4llm's own heuristics would flag as
    image-heavy (this fixture's embedded photo does exactly that; see
    `analyze_page`'s `img_text` reason).

    Patched at pymupdf4llm's OCR-selection entry point so the assertion
    holds regardless of whether tesseract/tessdata happens to be installed
    in the environment running the test.
    """
    doc = fitz.open(str(digital_bill_pdf))
    page = doc[CHART_PAGE_INDEX]
    assert page.get_text("text").strip()  # confirm this is the native-text case under test

    calls: list[None] = []

    def _spy_select_ocr_function():
        calls.append(None)
        return None

    monkeypatch.setattr(
        pymupdf4llm_document_layout, "select_ocr_function", _spy_select_ocr_function
    )

    pipeline._extract_page_markdown(doc, CHART_PAGE_INDEX)

    assert not calls, "OCR selection must never be reached for a page with a native text layer"
    doc.close()
