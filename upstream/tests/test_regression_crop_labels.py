"""Regression tests for the three bugs fixed in the chart-crop / label-filter pass.

Covered scenarios (all exercised via the bar_with_context synthetic fixture):

  1. x-axis label pollution  — unrelated page text below the chart must not
     appear in axes.x.labels.

  2. Chart-crop header bleed — a table header row placed just above the chart
     must not be pulled into chart_rect (i.e. chart_rect.y0 must stay below
     the header row).

  3. Missing highlighted current-month bar — the last bar, which uses a
     different fill color, must be found and merged into the main series.
"""

from __future__ import annotations

import fitz

from pdf_chart_parser.pipeline import extract_usage_chart
from pdf_chart_parser.vector.drawings import collect_drawings
from pdf_chart_parser.vector.locate import locate_chart
from pdf_chart_parser.vector.text import collect_text_spans

_MONTH_ABBREVS = {
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
}

# y-coordinate of the header row baseline in the synthetic fixture.
# Used to assert that chart_rect.y0 stays below it.
_HEADER_ROW_Y = 240.0


# ---------------------------------------------------------------------------
# Bug 1: x-axis label pollution
# ---------------------------------------------------------------------------


def test_x_labels_contain_only_month_names(bar_with_context_pdf):
    """axes.x.labels must contain only 3-letter month abbreviations."""
    result = extract_usage_chart(
        pdf_path=str(bar_with_context_pdf), return_annotated_image=False
    )
    labels = result["axes"]["x"]["labels"]

    bad = [lbl for lbl in labels if lbl not in _MONTH_ABBREVS]
    assert not bad, (
        f"Non-month labels leaked into x-axis (off-chart text pollution): {bad}"
    )


def test_x_labels_count(bar_with_context_pdf):
    """All 15 month labels must be present, no more."""
    result = extract_usage_chart(
        pdf_path=str(bar_with_context_pdf), return_annotated_image=False
    )
    labels = result["axes"]["x"]["labels"]
    assert len(labels) == 15, f"Expected 15 x-labels, got {len(labels)}: {labels}"


# ---------------------------------------------------------------------------
# Bug 2: chart crop header bleed
# ---------------------------------------------------------------------------


def test_chart_rect_does_not_include_header(bar_with_context_pdf):
    """chart_rect.y0 must stay below the header row above the chart."""
    doc = fitz.open(str(bar_with_context_pdf))
    page = doc[0]
    drawings = collect_drawings(page)
    spans = collect_text_spans(page)
    result = locate_chart(drawings, spans, "auto")
    doc.close()

    assert result is not None
    chart_rect = result[0]
    assert chart_rect.y0 > _HEADER_ROW_Y, (
        f"chart_rect.y0={chart_rect.y0:.1f} is above the header row "
        f"(baseline y≈{_HEADER_ROW_Y}); header bleed regression detected"
    )


# ---------------------------------------------------------------------------
# Bug 3: missing highlighted current-month bar
# ---------------------------------------------------------------------------


def test_all_15_bars_extracted(bar_with_context_pdf):
    """All 15 bars — including the differently-colored current-month bar — must
    be extracted."""
    result = extract_usage_chart(
        pdf_path=str(bar_with_context_pdf), return_annotated_image=False
    )
    total_bars = sum(len(s["points"]) for s in result["series"] if s["type"] == "bar")
    assert total_bars == 15, (
        f"Expected 15 bars (14 regular + 1 highlighted), got {total_bars}"
    )


def test_highlighted_bar_merged_into_single_series(bar_with_context_pdf):
    """The highlighted bar must be merged into the dominant series, not emitted
    as a separate 1-bar series."""
    result = extract_usage_chart(
        pdf_path=str(bar_with_context_pdf), return_annotated_image=False
    )
    bar_series = [s for s in result["series"] if s["type"] == "bar"]
    assert len(bar_series) == 1, (
        f"Expected 1 bar series (highlighted bar merged in), got {len(bar_series)}"
    )
    assert len(bar_series[0]["points"]) == 15


# ---------------------------------------------------------------------------
# Value accuracy
# ---------------------------------------------------------------------------


def test_bar_values_within_tolerance(bar_with_context_pdf, bar_with_context_expected):
    """Extracted values must match known ground-truth within tolerance."""
    result = extract_usage_chart(
        pdf_path=str(bar_with_context_pdf), return_annotated_image=False
    )
    bar_series = [s for s in result["series"] if s["type"] == "bar"]
    assert bar_series, "No bar series found"

    exp_pts = bar_with_context_expected["series"][0]["points"]
    act_pts = bar_series[0]["points"]
    tol = bar_with_context_expected["tolerance_pct"] / 100.0

    # Use positional comparison (months repeat in a 15-bar sliding window)
    n_pass = 0
    n_total = 0
    for act, exp in zip(act_pts, exp_pts):
        if exp["value"] == 0:
            continue
        n_total += 1
        if abs(act["value"] - exp["value"]) / abs(exp["value"]) <= tol:
            n_pass += 1

    assert n_total >= 13, f"Too few comparable points: {n_total}"
    assert n_pass >= int(n_total * 0.9), (
        f"Only {n_pass}/{n_total} bars within {tol * 100:.0f}% tolerance"
    )
