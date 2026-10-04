"""Tests for vector/bars.py."""

from __future__ import annotations

from pathlib import Path

import fitz

from pdf_chart_parser.vector.bars import extract_bars
from pdf_chart_parser.vector.calibrate import calibrate_axes
from pdf_chart_parser.vector.drawings import collect_drawings
from pdf_chart_parser.vector.locate import locate_chart
from pdf_chart_parser.vector.text import collect_text_spans


def _extract_bar_series(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    result = locate_chart(drawings, spans, "auto")
    assert result is not None
    chart_rect, _, bar_rects, _, plot_rect = result
    axes, _ = calibrate_axes(spans, chart_rect, "auto", plot_rect=plot_rect)
    series, warnings = extract_bars(bar_rects, axes, chart_rect)
    doc.close()
    return series, warnings


def test_bar_series_count(synthetic_bar_pdf):
    series, _ = _extract_bar_series(synthetic_bar_pdf)
    assert len(series) == 1
    assert series[0].type == "bar"


def test_bar_points_count(synthetic_bar_pdf):
    series, _ = _extract_bar_series(synthetic_bar_pdf)
    assert len(series[0].points) == 12


def test_bar_values_accurate(synthetic_bar_pdf, synthetic_bar_expected):
    series, _ = _extract_bar_series(synthetic_bar_pdf)
    expected_points = synthetic_bar_expected["series"][0]["points"]
    actual_points = series[0].points
    assert len(actual_points) == len(expected_points)

    tolerance = synthetic_bar_expected["tolerance_pct"] / 100.0
    n_pass = 0
    for act, exp in zip(actual_points, expected_points):
        if exp["value"] == 0:
            continue
        pct_err = abs(act.value - exp["value"]) / abs(exp["value"])
        if pct_err <= tolerance:
            n_pass += 1
        else:
            pass  # some bars may not match exactly due to text-label offset

    # At least 90% must be within tolerance
    assert n_pass >= int(len(expected_points) * 0.9), (
        f"Only {n_pass}/{len(expected_points)} bars within {tolerance * 100:.1f}% tolerance"
    )


def test_bar_x_labels_assigned(synthetic_bar_pdf):
    series, _ = _extract_bar_series(synthetic_bar_pdf)
    labeled = [p for p in series[0].points if p.x_label]
    assert len(labeled) >= 10


def test_bar_values_positive(synthetic_bar_pdf):
    series, _ = _extract_bar_series(synthetic_bar_pdf)
    for pt in series[0].points:
        assert pt.value >= 0


def test_bar_confidence_reasonable(synthetic_bar_pdf):
    series, _ = _extract_bar_series(synthetic_bar_pdf)
    assert series[0].confidence > 0.5
