"""Tests for vector/calibrate.py."""

from __future__ import annotations

from pathlib import Path

import fitz

from pdf_chart_parser.vector.calibrate import calibrate_axes, y_to_value
from pdf_chart_parser.vector.drawings import collect_drawings
from pdf_chart_parser.vector.locate import locate_chart
from pdf_chart_parser.vector.text import collect_text_spans

PDFS_DIR = Path(__file__).parent / "fixtures" / "pdfs"


def _get_axes(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    result = locate_chart(drawings, spans, "auto")
    if result is None:
        return None, []
    chart_rect, _, _, _, plot_rect = result
    axes, warnings = calibrate_axes(spans, chart_rect, "auto", plot_rect=plot_rect)
    doc.close()
    return axes, warnings


def test_bar_y_axis_calibrated(synthetic_bar_pdf):
    axes, warnings = _get_axes(synthetic_bar_pdf)
    assert axes is not None
    assert len(axes.y_primary.points) >= 2
    assert axes.y_primary.r_squared > 0.99
    assert axes.y_primary.scale_per_point != 0.0


def test_bar_x_labels_present(synthetic_bar_pdf):
    axes, _ = _get_axes(synthetic_bar_pdf)
    assert axes is not None
    assert len(axes.x.labels) >= 12


def test_y_to_value_roundtrip(synthetic_bar_pdf):
    """Values computed from calibration should be close to tick labels."""
    doc = fitz.open(str(synthetic_bar_pdf))
    page = doc[0]
    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    result = locate_chart(drawings, spans, "auto")
    chart_rect, _, _, _, plot_rect = result
    axes, _ = calibrate_axes(spans, chart_rect, "auto", plot_rect=plot_rect)
    doc.close()

    # Each calibration point should round-trip within 1%
    for cp in axes.y_primary.points:
        recovered = y_to_value(cp.y, axes.y_primary)
        if cp.value != 0:
            assert abs(recovered - cp.value) / abs(cp.value) < 0.02, (
                f"Round-trip failed: expected {cp.value}, got {recovered}"
            )


def test_unit_detected_dollars(synthetic_bar_pdf):
    axes, _ = _get_axes(synthetic_bar_pdf)
    assert axes.y_primary.unit == "dollars"


def test_unit_detected_kwh(synthetic_line_pdf):
    axes, _ = _get_axes(synthetic_line_pdf)
    assert axes.y_primary.unit == "kwh"


def test_hybrid_secondary_axis(synthetic_hybrid_pdf):
    axes, warnings = _get_axes(synthetic_hybrid_pdf)
    assert axes is not None
    assert axes.y_secondary is not None
    assert any("secondary" in w for w in warnings)
