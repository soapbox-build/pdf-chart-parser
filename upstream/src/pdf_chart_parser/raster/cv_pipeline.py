"""Raster CV/OCR fallback for scanned PDFs or failed vector extraction."""

from __future__ import annotations

import re
from typing import Any

import fitz
import numpy as np

from pdf_chart_parser.models import (
    Axes,
    AxisCalibration,
    AxisCalibrationPoint,
    AxisInfo,
    DataPoint,
    Series,
)
from pdf_chart_parser.raster.ocr import ocr_axis_labels, ocr_axis_values
from pdf_chart_parser.vector.text import collect_text_spans


def extract_raster(
    doc: Any,
    page_index: int,
    chart_type_hint: str,
    value_unit_hint: str,
    render_dpi: int,
    return_annotated_image: bool,
    warnings: list[str],
) -> dict[str, Any]:
    """Run OpenCV-based chart extraction as a fallback.

    Requires the [raster] extra: opencv-python-headless + pytesseract.
    cv2 is imported inside the function because it is an optional dependency.
    """
    import cv2

    page = doc[page_index]
    zoom = render_dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img_bytes = pix.tobytes("png")

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        warnings.append("cv2 failed to decode rendered page image")
        return _failed(warnings)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bar_candidates = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > w * 0.8 and h > 20 and w > 5:
            bar_candidates.append((x, y, w, h))

    if not bar_candidates:
        warnings.append("raster: no bar candidates found")
        return _failed(warnings)

    bar_candidates.sort(key=lambda b: b[0])

    h_img, w_img = img.shape[:2]
    # Prefer x-axis labels read from the page text layer (a real digital layer,
    # or one added by OCRmyPDF for scanned pages — both far more accurate than
    # cropping and OCR'ing a raw image strip). Fall back to strip OCR only when
    # the page has no usable text layer.
    bottom_labels = _text_layer_bottom_labels(page, zoom, h_img)
    if not bottom_labels:
        bottom_strip = img[h_img * 7 // 8 :, :]
        bottom_labels = ocr_axis_labels(bottom_strip)

    warnings.append("raster fallback used; accuracy may be lower than vector path")

    baseline_y = max(b[1] + b[3] for b in bar_candidates)

    # Calibrate the y-axis from left-axis tick labels so bar heights map to real
    # values. Prefer the page text layer (digital, or OCRmyPDF-added for scans);
    # both its values and y-positions are more reliable than strip OCR. Fall
    # back to OCR'ing the left image strip when no text layer is available.
    axis_pairs = _text_layer_y_axis_pairs(page, zoom, w_img)
    if len(axis_pairs) < 2:
        left_strip = img[:, : max(w_img // 8, 1)]
        axis_pairs = ocr_axis_values(left_strip)
    calib_points: list[AxisCalibrationPoint] = []
    scale_a: float | None = None
    intercept = 0.0
    r_squared = 0.0
    if len(axis_pairs) >= 2:
        ys = np.array([p[1] for p in axis_pairs])
        vals = np.array([p[0] for p in axis_pairs])
        a, b = np.polyfit(ys, vals, 1)
        predicted = a * ys + b
        ss_res = float(np.sum((vals - predicted) ** 2))
        ss_tot = float(np.sum((vals - vals.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
        scale_a, intercept = float(a), float(b)
        calib_points = [AxisCalibrationPoint(value=v, y=y) for v, y in axis_pairs]

    calibrated = scale_a is not None
    if calibrated:
        unit = value_unit_hint if value_unit_hint != "auto" else "auto"
        confidence = 0.6
    else:
        unit = "auto"
        confidence = 0.3
        warnings.append(
            "raster: y-axis could not be calibrated (OCR unavailable or no numeric "
            "ticks); bar values are unavailable, not real units"
        )

    # A low-confidence CV fit must not masquerade as usable data: only a
    # confident, calibrated fit is trusted enough to report real values.
    values_calibrated = confidence >= 0.5
    calibration_status = "calibrated" if values_calibrated else "low_confidence"

    series_points = []
    for i, (x, y, w, h) in enumerate(bar_candidates):
        bar_top_y = y
        value: float | None = None
        if values_calibrated:
            value = scale_a * bar_top_y - scale_a * baseline_y
            if value < 0:
                value = 0.0
            value = round(value, 4)
        x_label = bottom_labels[i] if i < len(bottom_labels) else str(i)
        series_points.append(
            DataPoint(
                x_label=x_label,
                x=float(x + w // 2),
                value=value,
                y=float(bar_top_y),
                baseline_y=float(baseline_y),
                confidence=confidence,
            )
        )

    bar_series = Series(
        id="s0",
        type="bar",
        label="",
        unit=unit,
        axis="y_primary",
        color=[0.3, 0.5, 0.8],
        confidence=confidence,
        points=series_points,
    )

    axes = Axes(
        x=AxisInfo(kind="categorical", labels=bottom_labels),
        y_primary=AxisCalibration(
            unit=unit,
            points=calib_points,
            scale_per_point=abs(scale_a) if calibrated else 0.0,
            intercept=intercept,
            scale_per_pixel=abs(scale_a) if calibrated else 0.0,
            r_squared=r_squared,
        ),
    )

    return {
        "chart_found": True,
        "method": "raster_cv",
        "chart_type": "bar",
        "axes": axes.model_dump(exclude_none=True),
        "series": [bar_series.model_dump()],
        "warnings": warnings,
        "confidence": confidence,
        "values_calibrated": values_calibrated,
        "calibration_status": calibration_status,
        "annotated_png": None,
    }


def _parse_numeric(token: str) -> float | None:
    """Parse a leading number from a tick label (drops $, commas, units)."""
    cleaned = token.strip().replace(",", "").replace("$", "")
    m = re.match(r"^(\d+(?:\.\d+)?)", cleaned)
    return float(m.group(1)) if m else None


def _text_layer_y_axis_pairs(page: Any, zoom: float, w_img: int) -> list[tuple[float, float]]:
    """Return (value, y_pixel) tick pairs from numeric spans in the left gutter.

    Reads the page's text layer (digital or OCRmyPDF-added) and keeps numeric
    labels sitting in the left ~1/6 of the page — the y-axis tick column. Span
    coordinates are in PDF points, so they are scaled by `zoom` to match the
    rendered pixel space the bars were detected in.
    """
    left_limit_px = max(w_img // 6, 1)
    pairs: list[tuple[float, float]] = []
    for span in collect_text_spans(page):
        if span.x_center * zoom > left_limit_px:
            continue
        value = _parse_numeric(span.text)
        if value is None:
            continue
        pairs.append((value, span.y_center * zoom))
    return pairs


def _text_layer_bottom_labels(page: Any, zoom: float, h_img: int) -> list[str]:
    """Return x-axis labels from text spans in the bottom 1/8 of the page.

    Labels are ordered left-to-right so they line up with the x-sorted bars.
    Span coordinates (PDF points) are scaled by `zoom` to match pixel space.
    """
    bottom_limit_px = h_img * 7 // 8
    bottom = [s for s in collect_text_spans(page) if s.y_center * zoom >= bottom_limit_px]
    bottom.sort(key=lambda s: s.x_center)
    return [s.text for s in bottom if s.text]


def _failed(warnings: list[str]) -> dict[str, Any]:
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
