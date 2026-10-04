"""Tests for vector/lines.py."""

from __future__ import annotations

from pathlib import Path

import fitz

from pdf_chart_parser.vector.calibrate import calibrate_axes
from pdf_chart_parser.vector.drawings import collect_drawings
from pdf_chart_parser.vector.lines import extract_lines
from pdf_chart_parser.vector.locate import locate_chart
from pdf_chart_parser.vector.text import collect_text_spans


def _extract_line_series(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    result = locate_chart(drawings, spans, "auto")
    assert result is not None
    chart_rect, _, _, line_paths, plot_rect = result
    axes, _ = calibrate_axes(spans, chart_rect, "auto", plot_rect=plot_rect)
    series, warnings = extract_lines(line_paths, axes, spans, chart_rect, plot_rect=plot_rect)
    doc.close()
    return series, warnings


def test_line_series_extracted(synthetic_line_pdf):
    series, _ = _extract_line_series(synthetic_line_pdf)
    assert len(series) >= 1
    assert series[0].type == "line"


def test_line_points_count(synthetic_line_pdf):
    series, _ = _extract_line_series(synthetic_line_pdf)
    # After dedup by x_label, should have at most 12 points
    assert len(series[0].points) <= 12
    assert len(series[0].points) >= 10


def test_line_values_accurate(synthetic_line_pdf, synthetic_line_expected):
    series, _ = _extract_line_series(synthetic_line_pdf)
    expected_pts = synthetic_line_expected["series"][0]["points"]
    actual_pts = series[0].points

    tolerance = synthetic_line_expected["tolerance_pct"] / 100.0
    # Build a map of x_label → expected value
    exp_map = {p["x_label"]: p["value"] for p in expected_pts}

    n_pass = 0
    n_checked = 0
    for act in actual_pts:
        if act.x_label in exp_map:
            exp_val = exp_map[act.x_label]
            if exp_val == 0:
                continue
            pct_err = abs(act.value - exp_val) / abs(exp_val)
            n_checked += 1
            if pct_err <= tolerance:
                n_pass += 1

    assert n_checked >= 8, f"Too few labeled points matched: {n_checked}"
    assert n_pass >= int(n_checked * 0.9), (
        f"Only {n_pass}/{n_checked} line points within {tolerance * 100:.1f}% tolerance"
    )


def test_hybrid_has_line_series(synthetic_hybrid_pdf):
    doc = fitz.open(str(synthetic_hybrid_pdf))
    page = doc[0]
    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    result = locate_chart(drawings, spans, "auto")
    assert result is not None
    chart_rect, _, _, line_paths, plot_rect = result
    axes, _ = calibrate_axes(spans, chart_rect, "auto", plot_rect=plot_rect)
    series, _ = extract_lines(line_paths, axes, spans, chart_rect, plot_rect=plot_rect)
    doc.close()
    assert len(series) >= 1
