"""Tests for raster CV/OCR fallback.

Skipped when opencv-python-headless or pytesseract is not installed.
"""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip(
    "cv2",
    reason="opencv-python-headless not installed; install pdf-chart-parser[raster] to enable",
)


def test_raster_extraction_runs(synthetic_bar_raster_pdf):
    from pdf_chart_parser.pipeline import extract_usage_chart

    result = extract_usage_chart(
        pdf_path=str(synthetic_bar_raster_pdf),
        return_annotated_image=False,
    )
    # Raster page has no text layer — should fall through to raster or fail gracefully
    assert result["method"] in ("raster_cv", "failed")


def test_raster_fallback_no_crash_on_missing_ocr(synthetic_bar_raster_pdf):
    """Even without tesseract, raster pipeline should not raise."""
    import numpy as np

    from pdf_chart_parser.raster.ocr import ocr_axis_labels

    dummy_img = np.zeros((50, 100, 3), dtype=np.uint8)
    labels = ocr_axis_labels(dummy_img)
    assert isinstance(labels, list)
