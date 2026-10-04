"""Regression tests for issues found during code review.

Each test targets a specific defect that was fixed, and is written to fail
against the pre-fix behavior.
"""

from __future__ import annotations

import fitz
import pytest

from pdf_chart_parser.models import (
    Axes,
    AxisCalibration,
    AxisCalibrationPoint,
    AxisInfo,
)
from pdf_chart_parser.pipeline import _compute_confidence, extract_usage_chart
from pdf_chart_parser.vector.bars import extract_bars
from pdf_chart_parser.vector.calibrate import calibrate_axes, y_to_value
from pdf_chart_parser.vector.color import cluster_by_color, quantize_color
from pdf_chart_parser.vector.drawings import RectItem, StrokedPath
from pdf_chart_parser.vector.lines import extract_lines
from pdf_chart_parser.vector.text import TextSpan, nearest_x_label


def _linear_axis(
    y_bottom: float, y_top: float, v_bottom: float, v_top: float, unit: str = "auto"
) -> AxisCalibration:
    """Build a calibrated axis mapping pixel y -> value (value = a*y + b)."""
    a = (v_top - v_bottom) / (y_top - y_bottom)
    b = v_bottom - a * y_bottom
    return AxisCalibration(
        unit=unit,
        points=[
            AxisCalibrationPoint(value=v_bottom, y=y_bottom),
            AxisCalibrationPoint(value=v_top, y=y_top),
            AxisCalibrationPoint(value=(v_bottom + v_top) / 2, y=(y_bottom + y_top) / 2),
        ],
        scale_per_point=a,
        intercept=b,
        scale_per_pixel=abs(a),
        r_squared=1.0,
    )


# --- Issue: input errors crash the MCP/pipeline boundary instead of failing cleanly ---


def test_bad_base64_returns_failed_result():
    result = extract_usage_chart(pdf_base64="@@@not-valid-base64@@@")
    assert result["chart_found"] is False
    assert result["method"] == "failed"
    assert any("failed to load" in w.lower() for w in result["warnings"])


def test_missing_path_returns_failed_result():
    result = extract_usage_chart(pdf_path="/nonexistent/does-not-exist.pdf")
    assert result["method"] == "failed"
    assert result["page"] == 0


def test_no_input_returns_failed_result():
    result = extract_usage_chart()
    assert result["method"] == "failed"


def test_multiple_inputs_returns_failed_result(synthetic_bar_pdf):
    result = extract_usage_chart(pdf_path=str(synthetic_bar_pdf), pdf_base64="x")
    assert result["method"] == "failed"


# --- Issue: uncalibrated y-axis yields all-zero values reported with high confidence ---


def test_uncalibrated_axis_is_low_confidence():
    axes = Axes(y_primary=AxisCalibration(points=[]))  # < 2 calibration points
    series = [_dummy_series()]
    assert _compute_confidence(axes, series, []) <= 0.2


def test_calibration_failure_emits_warning():
    plot = fitz.Rect(100, 100, 300, 400)
    # No parseable numeric tick labels anywhere.
    spans = [TextSpan(text="Usage", bbox=(60, 95, 95, 105))]
    _, warnings = calibrate_axes(spans, plot, "auto", plot_rect=plot)
    assert any("calibration failed" in w for w in warnings)


def _dummy_series():
    from pdf_chart_parser.models import DataPoint, Series

    return Series(
        id="s0",
        type="bar",
        points=[DataPoint(value=0.0, confidence=1.0)],
        confidence=1.0,
    )


# --- Issue: a 2-point right-axis fit (r2 == 1.0) spawns a phantom secondary axis ---


def _left_axis_spans() -> list[TextSpan]:
    return [
        TextSpan(text="0", bbox=(60, 395, 80, 405)),
        TextSpan(text="50", bbox=(60, 245, 85, 255)),
        TextSpan(text="100", bbox=(60, 95, 90, 105)),
    ]


def test_two_right_labels_do_not_create_secondary_axis():
    plot = fitz.Rect(100, 100, 300, 400)
    spans = _left_axis_spans() + [
        TextSpan(text="2024", bbox=(305, 395, 340, 405)),
        TextSpan(text="99", bbox=(305, 95, 330, 105)),
    ]
    axes, warnings = calibrate_axes(spans, plot, "auto", plot_rect=plot)
    assert axes.y_secondary is None
    assert not any("secondary" in w for w in warnings)


def test_three_right_labels_do_create_secondary_axis():
    plot = fitz.Rect(100, 100, 300, 400)
    spans = _left_axis_spans() + [
        TextSpan(text="200", bbox=(305, 395, 340, 405)),
        TextSpan(text="400", bbox=(305, 245, 340, 255)),
        TextSpan(text="600", bbox=(305, 95, 340, 105)),
    ]
    axes, warnings = calibrate_axes(spans, plot, "auto", plot_rect=plot)
    assert axes.y_secondary is not None
    assert any("secondary" in w for w in warnings)


# --- Issue: secondary axis never used when both axis units are "auto" ---


def _line_path() -> StrokedPath:
    pts = [fitz.Point(105, 350), fitz.Point(160, 300), fitz.Point(230, 250), fitz.Point(295, 200)]
    return StrokedPath(
        points=pts, stroke=(0.84, 0.19, 0.15), width=2.0, dashed=False, close_path=False, seqno=0
    )


def test_line_uses_secondary_axis_when_units_auto():
    plot = fitz.Rect(100, 100, 300, 400)
    axes = Axes(
        x=AxisInfo(labels=["A", "B", "C", "D"]),
        y_primary=_linear_axis(400, 100, 0, 100, unit="auto"),
        y_secondary=_linear_axis(400, 100, 0, 800, unit="auto"),
    )
    series, _ = extract_lines([_line_path()], axes, [], plot, plot_rect=plot)
    assert series
    # With both units "auto" the old guard left the line on the primary axis.
    assert series[0].axis == "y_secondary"


def test_line_stays_on_primary_when_units_known_and_equal():
    plot = fitz.Rect(100, 100, 300, 400)
    axes = Axes(
        x=AxisInfo(labels=["A", "B", "C", "D"]),
        y_primary=_linear_axis(400, 100, 0, 100, unit="kwh"),
        y_secondary=_linear_axis(400, 100, 0, 800, unit="kwh"),
    )
    series, _ = extract_lines([_line_path()], axes, [], plot, plot_rect=plot)
    assert series
    assert series[0].axis == "y_primary"


# --- Issue: bar x-labels assigned by ordinal index instead of x position ---


def test_bar_labels_assigned_by_position_with_a_gap():
    axes = Axes(
        x=AxisInfo(labels=["A", "B", "C", "D"]),
        y_primary=_linear_axis(400, 100, 0, 300),
    )
    fill = (0.22, 0.47, 0.72)
    # Three bars occupying label slots A, B, D (slot C has no bar).
    bars = [
        RectItem(rect=fitz.Rect(0, 200, 10, 400), fill=fill, stroke=None, width=10, seqno=0),
        RectItem(rect=fitz.Rect(20, 200, 30, 400), fill=fill, stroke=None, width=10, seqno=1),
        RectItem(rect=fitz.Rect(60, 200, 70, 400), fill=fill, stroke=None, width=10, seqno=2),
    ]
    series, _ = extract_bars(bars, axes, fitz.Rect(0, 100, 70, 400))
    labels = [p.x_label for p in series[0].points]
    # Position-based mapping skips C; the old index-based code produced A, B, C.
    assert labels == ["A", "B", "D"]


def test_nearest_x_label_maps_by_position():
    domain = fitz.Rect(0, 0, 100, 0)
    labels = ["A", "B", "C", "D", "E"]
    assert nearest_x_label(0, labels, domain) == "A"
    assert nearest_x_label(100, labels, domain) == "E"
    assert nearest_x_label(50, labels, domain) == "C"


# --- Issue: bar regrouping used a stricter metric than detection, splitting one series ---


def test_cluster_keeps_near_colors_together_across_quantize_boundary():
    a = (0.44, 0.50, 0.80)
    b = (0.46, 0.50, 0.80)
    # The two colors straddle a quantization bucket edge...
    assert quantize_color(a) != quantize_color(b)
    # ...but are within the Euclidean grouping threshold, so they stay one series.
    groups = cluster_by_color([a, b], lambda c: c)
    assert len(groups) == 1


# --- Issue: y_to_value re-fit a polynomial each call instead of reusing coefficients ---


def test_y_to_value_reuses_stored_coefficients():
    pts = [AxisCalibrationPoint(value=0.0, y=100.0), AxisCalibrationPoint(value=100.0, y=0.0)]
    cal = AxisCalibration(points=pts, scale_per_point=-1.0, intercept=100.0)
    assert y_to_value(50.0, cal) == pytest.approx(50.0)

    # Same coefficients but bogus points: result must come from the stored
    # slope/intercept, not a fresh fit over the points.
    cal_bogus_pts = AxisCalibration(
        points=[AxisCalibrationPoint(value=0.0, y=0.0), AxisCalibrationPoint(value=999.0, y=999.0)],
        scale_per_point=-1.0,
        intercept=100.0,
    )
    assert y_to_value(50.0, cal_bogus_pts) == pytest.approx(50.0)


def test_calibration_populates_intercept(synthetic_bar_pdf):
    doc = fitz.open(str(synthetic_bar_pdf))
    page = doc[0]
    from pdf_chart_parser.vector.drawings import collect_drawings
    from pdf_chart_parser.vector.locate import locate_chart
    from pdf_chart_parser.vector.text import collect_text_spans

    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    chart_rect, _, _, _, plot_rect = locate_chart(drawings, spans, "auto")
    axes, _ = calibrate_axes(spans, chart_rect, "auto", plot_rect=plot_rect)
    doc.close()

    # Every calibration point must round-trip through the stored slope/intercept.
    for cp in axes.y_primary.points:
        recovered = axes.y_primary.scale_per_point * cp.y + axes.y_primary.intercept
        assert recovered == pytest.approx(cp.value, abs=0.5)


# --- Issue: page_markdown described the whole document, not the selected page ---


def test_page_markdown_matches_selected_page(tmp_path):
    doc = fitz.open()
    p0 = doc.new_page(width=612, height=792)
    p0.insert_text(fitz.Point(72, 72), "COVER PAGE ALPHA")
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text(fitz.Point(72, 72), "Monthly usage in kWh and billing charges BETA")
    pdf_path = tmp_path / "two_page.pdf"
    doc.save(str(pdf_path))
    doc.close()

    result = extract_usage_chart(pdf_path=str(pdf_path), return_annotated_image=False)
    assert result["page"] == 2  # 1-based: BETA is the second page
    assert "BETA" in result["page_markdown"]
    assert "ALPHA" not in result["page_markdown"]


# --- Issue: raster fallback reported raw pixel heights as if calibrated values ---


def test_raster_uncalibrated_path_is_honest(synthetic_bar_raster_pdf, monkeypatch):
    pytest.importorskip(
        "cv2", reason="opencv-python-headless not installed; install pdf-chart-parser[raster]"
    )
    import pdf_chart_parser.raster.cv_pipeline as cvp

    monkeypatch.setattr(cvp, "ocr_axis_values", lambda img: [])
    monkeypatch.setattr(cvp, "ocr_axis_labels", lambda img: [])
    # Also neutralize the text-layer source so no axis values are available from
    # any source — this is what "uncalibrated" must look like.
    monkeypatch.setattr(cvp, "_text_layer_y_axis_pairs", lambda *a, **k: [])
    monkeypatch.setattr(cvp, "_text_layer_bottom_labels", lambda *a, **k: [])

    result = extract_usage_chart(
        pdf_path=str(synthetic_bar_raster_pdf), return_annotated_image=False
    )
    if result["method"] != "raster_cv":
        pytest.skip("raster path not exercised on this fixture")
    assert result["axes"]["y_primary"]["unit"] == "auto"
    assert result["confidence"] <= 0.3
    assert any("could not be calibrated" in w for w in result["warnings"])


def test_raster_calibrates_from_ocr_values(synthetic_bar_raster_pdf, monkeypatch):
    pytest.importorskip(
        "cv2", reason="opencv-python-headless not installed; install pdf-chart-parser[raster]"
    )
    import pdf_chart_parser.raster.cv_pipeline as cvp

    # value increases as pixel y decreases (top of image = larger value)
    monkeypatch.setattr(
        cvp, "ocr_axis_values", lambda img: [(0.0, 900.0), (100.0, 500.0), (200.0, 100.0)]
    )
    monkeypatch.setattr(cvp, "ocr_axis_labels", lambda img: [])
    # Disable the text-layer source so the strip-OCR calibration path under test
    # is the one actually exercised.
    monkeypatch.setattr(cvp, "_text_layer_y_axis_pairs", lambda *a, **k: [])
    monkeypatch.setattr(cvp, "_text_layer_bottom_labels", lambda *a, **k: [])

    result = extract_usage_chart(
        pdf_path=str(synthetic_bar_raster_pdf), return_annotated_image=False
    )
    if result["method"] != "raster_cv":
        pytest.skip("raster path not exercised on this fixture")
    assert result["axes"]["y_primary"]["r_squared"] > 0.99
    assert result["axes"]["y_primary"]["scale_per_point"] != 0.0
    points = result["series"][0]["points"]
    assert points
    # Calibrated values are bounded by the axis range, not raw pixel heights.
    assert all(0.0 <= p["value"] <= 400.0 for p in points)
