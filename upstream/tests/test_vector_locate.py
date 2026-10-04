"""Tests for vector/locate.py."""

from __future__ import annotations

from pathlib import Path

import fitz

from pdf_chart_parser.vector.color import color_saturation
from pdf_chart_parser.vector.drawings import collect_drawings
from pdf_chart_parser.vector.locate import _is_axis_or_gridline, locate_chart
from pdf_chart_parser.vector.text import collect_text_spans

PDFS_DIR = Path(__file__).parent / "fixtures" / "pdfs"


def _open_first_page(pdf_path: Path) -> fitz.Page:
    doc = fitz.open(str(pdf_path))
    return doc[0]


def test_locate_bar_chart(synthetic_bar_pdf):
    page = _open_first_page(synthetic_bar_pdf)
    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    result = locate_chart(drawings, spans, "auto")
    assert result is not None
    chart_rect, chart_type, bar_rects, line_paths, plot_rect = result
    assert chart_type == "bar"
    assert len(bar_rects) >= 12
    assert not line_paths


def test_locate_line_chart(synthetic_line_pdf):
    page = _open_first_page(synthetic_line_pdf)
    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    result = locate_chart(drawings, spans, "auto")
    assert result is not None
    chart_rect, chart_type, bar_rects, line_paths, plot_rect = result
    assert chart_type == "line"
    assert not bar_rects
    assert len(line_paths) >= 1


def test_locate_hybrid_chart(synthetic_hybrid_pdf):
    page = _open_first_page(synthetic_hybrid_pdf)
    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    result = locate_chart(drawings, spans, "auto")
    assert result is not None
    chart_rect, chart_type, bar_rects, line_paths, plot_rect = result
    assert chart_type == "hybrid"
    assert len(bar_rects) >= 10
    assert len(line_paths) >= 1


def test_axis_or_gridline_horizontal():
    import fitz

    from pdf_chart_parser.vector.drawings import StrokedPath

    # Purely horizontal path — should be detected as axis/gridline
    pts = [fitz.Point(50, 200), fitz.Point(500, 200)]
    path = StrokedPath(
        points=pts, stroke=(0.5, 0.5, 0.5), width=1.0, dashed=False, close_path=False, seqno=0
    )
    assert _is_axis_or_gridline(path)


def test_axis_or_gridline_data_line():
    import fitz

    from pdf_chart_parser.vector.drawings import StrokedPath

    # Diagonal colored path — not an axis
    pts = [fitz.Point(50, 300), fitz.Point(100, 250), fitz.Point(150, 280), fitz.Point(200, 220)]
    path = StrokedPath(
        points=pts, stroke=(0.85, 0.1, 0.1), width=2.0, dashed=False, close_path=False, seqno=0
    )
    assert not _is_axis_or_gridline(path)


def test_color_saturation_saturated():
    assert color_saturation((0.9, 0.1, 0.1)) > 0.5


def test_color_saturation_gray():
    assert color_saturation((0.5, 0.5, 0.5)) < 0.01


def test_hint_bar_suppresses_lines(synthetic_hybrid_pdf):
    page = _open_first_page(synthetic_hybrid_pdf)
    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    result = locate_chart(drawings, spans, "bar")
    if result is not None:
        _, chart_type, _, line_paths, _ = result
        assert chart_type == "bar"
        assert not line_paths
