"""Tests for the values_calibrated / calibration_status contract.

A chart's bars or lines can be geometrically detected even when there is no
way to know what real value they represent (no numeric y-axis, or a
low-confidence fit). These tests assert that callers get an explicit,
checkable signal for that instead of having to infer it from confidence
scores or spot fabricated zeros.
"""

from __future__ import annotations

import fitz
import pytest

from pdf_chart_parser.models import Axes, AxisCalibration, AxisCalibrationPoint, AxisInfo
from pdf_chart_parser.pipeline import _calibration_status, extract_usage_chart
from pdf_chart_parser.vector.bars import extract_bars
from pdf_chart_parser.vector.calibrate import is_axis_calibrated
from pdf_chart_parser.vector.drawings import RectItem, StrokedPath
from pdf_chart_parser.vector.lines import extract_lines

_UNCALIBRATED_AXIS = AxisCalibration(points=[])  # fewer than 2 tick points
_CALIBRATED_AXIS = AxisCalibration(
    points=[
        AxisCalibrationPoint(value=0.0, y=400.0),
        AxisCalibrationPoint(value=100.0, y=200.0),
    ],
    scale_per_point=-0.5,
    intercept=200.0,
    scale_per_pixel=0.5,
    r_squared=0.999,
)


# ---------------------------------------------------------------------------
# End-to-end: vector path
# ---------------------------------------------------------------------------


def test_uncalibrated_chart_reports_no_values(synthetic_bar_no_axis_pdf):
    """A chart with bars but no numeric y-axis must not report fabricated data."""
    result = extract_usage_chart(
        pdf_path=str(synthetic_bar_no_axis_pdf), return_annotated_image=False
    )
    assert result["chart_found"] is True
    assert result["values_calibrated"] is False
    assert result["calibration_status"] == "uncalibrated_axis"

    points = [p for s in result["series"] for p in s["points"]]
    assert points, "expected bars to still be detected"
    assert all(p["value"] is None for p in points), (
        "uncalibrated points must be null, never a fabricated number"
    )


def test_calibrated_chart_reports_real_values(synthetic_bar_pdf):
    result = extract_usage_chart(pdf_path=str(synthetic_bar_pdf), return_annotated_image=False)
    assert result["values_calibrated"] is True
    assert result["calibration_status"] == "calibrated"

    points = [p for s in result["series"] for p in s["points"]]
    assert points
    assert all(isinstance(p["value"], float) for p in points)


def test_printed_value_labels_count_as_calibrated(bar_with_context_pdf):
    """A chart with no numeric axis but real per-bar printed labels is not the
    same failure mode as a bill with neither — the values are genuine, just
    sourced differently, so they must not be nulled out."""
    result = extract_usage_chart(
        pdf_path=str(bar_with_context_pdf), return_annotated_image=False
    )
    assert result["values_calibrated"] is True
    assert result["calibration_status"] == "calibrated"

    points = [p for s in result["series"] for p in s["points"]]
    assert points
    assert all(p["value"] is not None for p in points)


def test_annotation_does_not_crash_on_uncalibrated_chart(synthetic_bar_no_axis_pdf):
    result = extract_usage_chart(
        pdf_path=str(synthetic_bar_no_axis_pdf), return_annotated_image=True
    )
    assert result["annotated_png"] is not None
    assert result["annotated_png"][:4] == b"\x89PNG"


def test_failed_result_is_no_chart():
    result = extract_usage_chart(pdf_path="/nonexistent/does-not-exist.pdf")
    assert result["values_calibrated"] is False
    assert result["calibration_status"] == "no_chart"


# ---------------------------------------------------------------------------
# End-to-end: raster path
# ---------------------------------------------------------------------------


def test_raster_low_confidence_is_uncalibrated(synthetic_bar_raster_pdf, monkeypatch):
    pytest.importorskip(
        "cv2", reason="opencv-python-headless not installed; install pdf-chart-parser[raster]"
    )
    import pdf_chart_parser.raster.cv_pipeline as cvp

    monkeypatch.setattr(cvp, "ocr_axis_values", lambda img: [])
    monkeypatch.setattr(cvp, "ocr_axis_labels", lambda img: [])
    monkeypatch.setattr(cvp, "_text_layer_y_axis_pairs", lambda *a, **k: [])
    monkeypatch.setattr(cvp, "_text_layer_bottom_labels", lambda *a, **k: [])

    result = extract_usage_chart(
        pdf_path=str(synthetic_bar_raster_pdf), return_annotated_image=False
    )
    if result["method"] != "raster_cv":
        pytest.skip("raster path not exercised on this fixture")

    assert result["values_calibrated"] is False
    assert result["calibration_status"] == "low_confidence"
    points = [p for s in result["series"] for p in s["points"]]
    assert points
    assert all(p["value"] is None for p in points)


def test_raster_calibrated_reports_real_values(synthetic_bar_raster_pdf, monkeypatch):
    pytest.importorskip(
        "cv2", reason="opencv-python-headless not installed; install pdf-chart-parser[raster]"
    )
    import pdf_chart_parser.raster.cv_pipeline as cvp

    monkeypatch.setattr(
        cvp, "ocr_axis_values", lambda img: [(0.0, 900.0), (100.0, 500.0), (200.0, 100.0)]
    )
    monkeypatch.setattr(cvp, "ocr_axis_labels", lambda img: [])
    monkeypatch.setattr(cvp, "_text_layer_y_axis_pairs", lambda *a, **k: [])
    monkeypatch.setattr(cvp, "_text_layer_bottom_labels", lambda *a, **k: [])

    result = extract_usage_chart(
        pdf_path=str(synthetic_bar_raster_pdf), return_annotated_image=False
    )
    if result["method"] != "raster_cv":
        pytest.skip("raster path not exercised on this fixture")

    assert result["values_calibrated"] is True
    assert result["calibration_status"] == "calibrated"
    points = [p for s in result["series"] for p in s["points"]]
    assert points
    assert all(p["value"] is not None for p in points)


# ---------------------------------------------------------------------------
# Unit: is_axis_calibrated
# ---------------------------------------------------------------------------


def test_is_axis_calibrated_requires_two_points_and_good_fit():
    assert is_axis_calibrated(_CALIBRATED_AXIS) is True
    assert is_axis_calibrated(_UNCALIBRATED_AXIS) is False

    single_point = AxisCalibration(points=[AxisCalibrationPoint(value=0.0, y=0.0)], r_squared=1.0)
    assert is_axis_calibrated(single_point) is False

    poor_fit = AxisCalibration(
        points=[
            AxisCalibrationPoint(value=0.0, y=400.0),
            AxisCalibrationPoint(value=100.0, y=200.0),
        ],
        r_squared=0.5,
    )
    assert is_axis_calibrated(poor_fit) is False


# ---------------------------------------------------------------------------
# Unit: extract_bars / extract_lines null out untrustworthy values
# ---------------------------------------------------------------------------


def test_bars_null_value_when_uncalibrated_and_no_labels():
    axes = Axes(x=AxisInfo(labels=["A", "B"]), y_primary=_UNCALIBRATED_AXIS)
    fill = (0.22, 0.47, 0.72)
    bars = [
        RectItem(rect=fitz.Rect(0, 200, 10, 400), fill=fill, stroke=None, width=10, seqno=0),
        RectItem(rect=fitz.Rect(20, 150, 30, 400), fill=fill, stroke=None, width=10, seqno=1),
    ]
    series, _ = extract_bars(bars, axes, fitz.Rect(0, 100, 30, 400))
    assert len(series[0].points) == 2
    assert all(p.value is None for p in series[0].points)
    # Detected geometry (position/height) must still be present for debugging.
    assert all(p.y is not None and p.baseline_y is not None for p in series[0].points)


def test_bars_keep_value_when_calibrated():
    axes = Axes(x=AxisInfo(labels=["A", "B"]), y_primary=_CALIBRATED_AXIS)
    fill = (0.22, 0.47, 0.72)
    bars = [
        RectItem(rect=fitz.Rect(0, 200, 10, 400), fill=fill, stroke=None, width=10, seqno=0),
    ]
    series, _ = extract_bars(bars, axes, fitz.Rect(0, 100, 30, 400))
    assert series[0].points[0].value is not None


def test_lines_null_value_when_uncalibrated():
    axes = Axes(x=AxisInfo(labels=["A", "B", "C", "D"]), y_primary=_UNCALIBRATED_AXIS)
    plot = fitz.Rect(100, 100, 300, 400)
    path = StrokedPath(
        points=[fitz.Point(105, 350), fitz.Point(295, 200)],
        stroke=(0.84, 0.19, 0.15),
        width=2.0,
        dashed=False,
        close_path=False,
        seqno=0,
    )
    series, _ = extract_lines([path], axes, [], plot, plot_rect=plot)
    assert series
    assert all(p.value is None for p in series[0].points)


def test_lines_keep_value_when_calibrated():
    axes = Axes(x=AxisInfo(labels=["A", "B", "C", "D"]), y_primary=_CALIBRATED_AXIS)
    plot = fitz.Rect(100, 100, 300, 400)
    path = StrokedPath(
        points=[fitz.Point(105, 350), fitz.Point(295, 200)],
        stroke=(0.84, 0.19, 0.15),
        width=2.0,
        dashed=False,
        close_path=False,
        seqno=0,
    )
    series, _ = extract_lines([path], axes, [], plot, plot_rect=plot)
    assert series
    assert all(p.value is not None for p in series[0].points)


# ---------------------------------------------------------------------------
# Unit: _calibration_status
# ---------------------------------------------------------------------------


def test_calibration_status_branches():
    from pdf_chart_parser.models import DataPoint, Series

    calibrated_series = [Series(id="s0", type="bar", points=[DataPoint(value=1.0)])]
    assert _calibration_status(Axes(y_primary=_CALIBRATED_AXIS), calibrated_series) == (
        True,
        "calibrated",
    )

    null_series = [Series(id="s0", type="bar", points=[DataPoint(value=None)])]
    assert _calibration_status(Axes(y_primary=_UNCALIBRATED_AXIS), null_series) == (
        False,
        "uncalibrated_axis",
    )

    # Axis found >= 2 tick points but a point still ended up null (e.g. a
    # secondary-axis line where only the primary axis fit well).
    partially_fit_axis = AxisCalibration(
        points=[
            AxisCalibrationPoint(value=0.0, y=400.0),
            AxisCalibrationPoint(value=100.0, y=200.0),
        ],
        r_squared=0.999,
    )
    assert _calibration_status(Axes(y_primary=partially_fit_axis), null_series) == (
        False,
        "low_confidence",
    )
