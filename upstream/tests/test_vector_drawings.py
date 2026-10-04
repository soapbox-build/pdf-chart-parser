"""Tests for vector/drawings.py."""

from __future__ import annotations

import fitz

from pdf_chart_parser.vector.drawings import RectItem, StrokedPath, collect_drawings


def _make_page_with_bar_chart() -> fitz.Page:
    """Helper: open the synthetic bar PDF and return its first page."""
    from pathlib import Path

    pdf_path = Path(__file__).parent / "fixtures" / "pdfs" / "synthetic" / "bar.pdf"
    doc = fitz.open(str(pdf_path))
    return doc[0]


def test_collect_drawings_returns_dicts():
    page = _make_page_with_bar_chart()
    result = collect_drawings(page)
    assert "rects" in result
    assert "paths" in result


def test_bar_chart_has_fill_rects():
    page = _make_page_with_bar_chart()
    result = collect_drawings(page)
    fill_rects = [r for r in result["rects"] if r.fill is not None]
    assert len(fill_rects) >= 12, "Expected at least 12 bar rects"


def test_rect_items_have_valid_fields():
    page = _make_page_with_bar_chart()
    result = collect_drawings(page)
    for r in result["rects"]:
        assert isinstance(r, RectItem)
        assert r.rect.width > 0
        assert r.rect.height > 0


def test_stroked_paths_have_points():
    from pathlib import Path

    pdf_path = Path(__file__).parent / "fixtures" / "pdfs" / "synthetic" / "line.pdf"
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    result = collect_drawings(page)
    # Line chart should have stroked paths
    paths = result["paths"]
    assert len(paths) > 0
    for p in paths:
        assert isinstance(p, StrokedPath)
        assert len(p.points) >= 2
