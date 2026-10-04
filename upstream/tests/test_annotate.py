"""Tests for annotate.py."""

from __future__ import annotations

import fitz

from pdf_chart_parser.annotate import annotate_chart
from pdf_chart_parser.models import Axes, AxisCalibration, AxisCalibrationPoint, AxisInfo
from pdf_chart_parser.vector.bars import extract_bars
from pdf_chart_parser.vector.calibrate import calibrate_axes
from pdf_chart_parser.vector.drawings import collect_drawings
from pdf_chart_parser.vector.locate import locate_chart
from pdf_chart_parser.vector.text import collect_text_spans


def test_annotate_produces_png(synthetic_bar_pdf):
    doc = fitz.open(str(synthetic_bar_pdf))
    page = doc[0]
    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    result = locate_chart(drawings, spans, "auto")
    assert result is not None
    chart_rect, chart_type, bar_rects, _, plot_rect = result
    axes, _ = calibrate_axes(spans, chart_rect, "auto", plot_rect=plot_rect)
    series, _ = extract_bars(bar_rects, axes, chart_rect)

    png_bytes = annotate_chart(page, chart_rect, series, axes, render_dpi=72, chart_type=chart_type)
    doc.close()

    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000
    assert png_bytes[:4] == b"\x89PNG"


def test_annotate_with_empty_series(synthetic_bar_pdf):
    doc = fitz.open(str(synthetic_bar_pdf))
    page = doc[0]
    axes = Axes(
        x=AxisInfo(kind="categorical", labels=[]),
        y_primary=AxisCalibration(
            unit="dollars",
            points=[AxisCalibrationPoint(value=0, y=400), AxisCalibrationPoint(value=100, y=200)],
            scale_per_point=-0.5,
            scale_per_pixel=0.2,
            r_squared=0.999,
        ),
    )
    png_bytes = annotate_chart(page, fitz.Rect(80, 150, 540, 550), [], axes, render_dpi=72)
    doc.close()
    assert png_bytes[:4] == b"\x89PNG"
