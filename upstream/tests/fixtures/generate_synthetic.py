"""Generate synthetic born-digital PDFs for testing.

Run: uv run python tests/fixtures/generate_synthetic.py
Produces: tests/fixtures/pdfs/synthetic/*.pdf and tests/fixtures/expected/synthetic/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent
PDFS_DIR = FIXTURES_DIR / "pdfs" / "synthetic"
EXPECTED_DIR = FIXTURES_DIR / "expected" / "synthetic"

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Known values for reproducibility
BAR_VALUES = [120.0, 135.5, 98.0, 145.2, 160.8, 175.0, 188.5, 172.3, 150.0, 130.7, 115.0, 140.0]

# 15-month sliding window (MAR prior-year through MAY current-year) for the
# utility-context chart.  Last bar uses a different fill color (highlighted
# current month) to exercise the baseline-absorption fix.
CONTEXT_MONTHS = [
    "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP",
    "OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY",
]
CONTEXT_VALUES = [
    50.0, 40.0, 40.0, 90.0, 190.0, 210.0, 200.0,
    115.0, 60.0, 45.0, 45.0, 50.0, 40.0, 45.0, 50.0,
]
LINE_VALUES = [55.1, 62.3, 78.9, 85.0, 90.2, 95.5, 88.7, 82.1, 75.3, 68.0, 60.4, 58.8]
HYBRID_BAR_VALUES = [
    210.0,
    225.0,
    190.0,
    240.0,
    260.0,
    280.0,
    295.0,
    275.0,
    250.0,
    230.0,
    205.0,
    220.0,
]
HYBRID_LINE_VALUES = [
    520.0,
    580.0,
    650.0,
    700.0,
    720.0,
    750.0,
    730.0,
    690.0,
    640.0,
    600.0,
    555.0,
    510.0,
]


def _make_bar_pdf(path: Path) -> None:
    """Create a simple bar chart PDF with known dollar values."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    chart_left = 80.0
    chart_right = 540.0
    chart_top = 150.0
    chart_bottom = 550.0
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top

    max_val = max(BAR_VALUES)
    n = len(BAR_VALUES)
    bar_width = chart_w / (n * 1.5)
    bar_spacing = chart_w / n

    # Y-axis tick marks and labels (left side)
    y_ticks = [0, 50, 100, 150, 200]
    for tick in y_ticks:
        y_px = chart_bottom - (tick / max_val) * chart_h
        # Tick mark
        page.draw_line(fitz.Point(chart_left - 5, y_px), fitz.Point(chart_left, y_px))
        # Label
        page.insert_text(
            fitz.Point(chart_left - 40, y_px + 4),
            f"${tick}",
            fontsize=9,
            color=(0, 0, 0),
        )
        # Gridline
        page.draw_line(
            fitz.Point(chart_left, y_px),
            fitz.Point(chart_right, y_px),
            color=(0.8, 0.8, 0.8),
            width=0.5,
        )

    # Axis lines
    page.draw_line(
        fitz.Point(chart_left, chart_top), fitz.Point(chart_left, chart_bottom), width=1.5
    )
    page.draw_line(
        fitz.Point(chart_left, chart_bottom), fitz.Point(chart_right, chart_bottom), width=1.5
    )

    # Bars
    bar_fill = (0.22, 0.47, 0.72)
    for i, val in enumerate(BAR_VALUES):
        bar_top = chart_bottom - (val / max_val) * chart_h
        bar_x0 = chart_left + i * bar_spacing + (bar_spacing - bar_width) / 2
        bar_x1 = bar_x0 + bar_width
        page.draw_rect(
            fitz.Rect(bar_x0, bar_top, bar_x1, chart_bottom),
            fill=bar_fill,
            color=None,
        )
        # X label
        page.insert_text(
            fitz.Point(bar_x0 + bar_width / 2 - 8, chart_bottom + 12),
            MONTHS[i],
            fontsize=8,
            color=(0, 0, 0),
        )

    # Title
    page.insert_text(fitz.Point(200, 130), "Monthly Utility Charges ($)", fontsize=12)

    doc.save(str(path))
    doc.close()


def _make_line_pdf(path: Path) -> None:
    """Create a line chart PDF with known kWh usage values."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    chart_left = 80.0
    chart_right = 540.0
    chart_top = 150.0
    chart_bottom = 550.0
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top

    max_val = 120.0
    n = len(LINE_VALUES)
    x_spacing = chart_w / (n - 1)

    # Y-axis
    y_ticks = [0, 25, 50, 75, 100, 120]
    for tick in y_ticks:
        y_px = chart_bottom - (tick / max_val) * chart_h
        page.draw_line(fitz.Point(chart_left - 5, y_px), fitz.Point(chart_left, y_px))
        page.insert_text(
            fitz.Point(chart_left - 40, y_px + 4),
            f"{tick} kWh",
            fontsize=9,
            color=(0, 0, 0),
        )
        page.draw_line(
            fitz.Point(chart_left, y_px),
            fitz.Point(chart_right, y_px),
            color=(0.8, 0.8, 0.8),
            width=0.5,
        )

    # Axes
    page.draw_line(
        fitz.Point(chart_left, chart_top), fitz.Point(chart_left, chart_bottom), width=1.5
    )
    page.draw_line(
        fitz.Point(chart_left, chart_bottom), fitz.Point(chart_right, chart_bottom), width=1.5
    )

    # Line series
    line_color = (0.85, 0.33, 0.10)
    pts = []
    for i, val in enumerate(LINE_VALUES):
        x = chart_left + i * x_spacing
        y = chart_bottom - (val / max_val) * chart_h
        pts.append(fitz.Point(x, y))
        page.insert_text(
            fitz.Point(x - 8, chart_bottom + 12),
            MONTHS[i],
            fontsize=8,
            color=(0, 0, 0),
        )

    for i in range(len(pts) - 1):
        page.draw_line(pts[i], pts[i + 1], color=line_color, width=2.0)

    # Marker dots
    for pt in pts:
        page.draw_circle(pt, 3, color=line_color, fill=line_color)

    page.insert_text(fitz.Point(200, 130), "Monthly Energy Usage (kWh)", fontsize=12)
    doc.save(str(path))
    doc.close()


def _make_hybrid_pdf(path: Path) -> None:
    """Create a hybrid bar+line chart with dual y-axis."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    chart_left = 80.0
    chart_right = 540.0
    chart_top = 150.0
    chart_bottom = 550.0
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top

    max_bar = 320.0
    max_line = 800.0
    n = len(HYBRID_BAR_VALUES)
    bar_width = chart_w / (n * 1.5)
    bar_spacing = chart_w / n

    # Left Y-axis (dollars)
    for tick in [0, 100, 200, 300]:
        y_px = chart_bottom - (tick / max_bar) * chart_h
        page.draw_line(fitz.Point(chart_left - 5, y_px), fitz.Point(chart_left, y_px))
        page.insert_text(
            fitz.Point(chart_left - 45, y_px + 4),
            f"${tick}",
            fontsize=9,
            color=(0, 0, 0),
        )
        page.draw_line(
            fitz.Point(chart_left, y_px),
            fitz.Point(chart_right, y_px),
            color=(0.85, 0.85, 0.85),
            width=0.5,
        )

    # Right Y-axis (kWh)
    for tick in [0, 200, 400, 600, 800]:
        y_px = chart_bottom - (tick / max_line) * chart_h
        page.draw_line(fitz.Point(chart_right, y_px), fitz.Point(chart_right + 5, y_px))
        page.insert_text(
            fitz.Point(chart_right + 8, y_px + 4),
            f"{tick} kWh",
            fontsize=9,
            color=(0, 0, 0),
        )

    # Axis lines
    page.draw_line(
        fitz.Point(chart_left, chart_top), fitz.Point(chart_left, chart_bottom), width=1.5
    )
    page.draw_line(
        fitz.Point(chart_left, chart_bottom), fitz.Point(chart_right, chart_bottom), width=1.5
    )
    page.draw_line(
        fitz.Point(chart_right, chart_top), fitz.Point(chart_right, chart_bottom), width=1.5
    )

    # Bars (charges)
    bar_fill = (0.22, 0.47, 0.72)
    for i, val in enumerate(HYBRID_BAR_VALUES):
        bar_top = chart_bottom - (val / max_bar) * chart_h
        bar_x0 = chart_left + i * bar_spacing + (bar_spacing - bar_width) / 2
        bar_x1 = bar_x0 + bar_width
        page.draw_rect(
            fitz.Rect(bar_x0, bar_top, bar_x1, chart_bottom),
            fill=bar_fill,
            color=None,
        )
        page.insert_text(
            fitz.Point(bar_x0 + bar_width / 2 - 8, chart_bottom + 12),
            MONTHS[i],
            fontsize=8,
            color=(0, 0, 0),
        )

    # Line series (kWh on secondary axis)
    line_color = (0.84, 0.19, 0.15)
    pts = []
    x_spacing = chart_w / (n - 1)
    for i, val in enumerate(HYBRID_LINE_VALUES):
        x = chart_left + i * x_spacing
        y = chart_bottom - (val / max_line) * chart_h
        pts.append(fitz.Point(x, y))

    for i in range(len(pts) - 1):
        page.draw_line(pts[i], pts[i + 1], color=line_color, width=2.0)

    for pt in pts:
        page.draw_circle(pt, 3, color=line_color, fill=line_color)

    page.insert_text(fitz.Point(180, 130), "Monthly Charges ($) and Usage (kWh)", fontsize=12)
    doc.save(str(path))
    doc.close()


def _make_bar_with_context_pdf(path: Path) -> None:
    """Bar chart that exercises three previously-regressed behaviours:

    1. A table-header row placed ~30 pt above the chart (within the old uniform
       60-pt search margin, but outside the new 20-pt top margin).  Verifies
       that chart_rect.y0 does not extend into the header.

    2. 15 bars where the last one uses a different fill color (the "highlighted
       current month" pattern).  Verifies that the differently-colored bar is
       absorbed into the main series rather than silently dropped.

    3. Unrelated section text ~80 pt below the chart baseline (further than the
       new chart_rect.y1 + 10 pt upper bound).  Verifies that those spans are
       not captured as x-axis labels.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    chart_left = 100.0
    chart_right = 560.0
    chart_top = 270.0
    chart_bottom = 430.0
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top
    max_val = 240.0
    n = len(CONTEXT_VALUES)
    bar_width = chart_w / (n * 1.5)
    bar_spacing = chart_w / n

    # Header row ~30 pt above chart_top.  Its y_center (~237) is above the new
    # 20-pt top search window (y0_search = core.y0 - 20 ≈ 270) but inside the
    # old 60-pt window (y0_search_old = core.y0 - 60 ≈ 230).
    header_y = chart_top - 30  # baseline at 240, y_center ≈ 237
    page.insert_text(fitz.Point(chart_left, header_y), "Usage History", fontsize=8)
    page.insert_text(fitz.Point(chart_left + 90, header_y), "SERVICE ADDRESS", fontsize=8)
    page.insert_text(fitz.Point(chart_left + 200, header_y), "METER", fontsize=8)

    # Y-axis tick labels and gridlines
    for tick in [0, 100, 200]:
        y_px = chart_bottom - (tick / max_val) * chart_h
        page.draw_line(fitz.Point(chart_left - 5, y_px), fitz.Point(chart_left, y_px))
        page.insert_text(fitz.Point(chart_left - 40, y_px + 4), f"${tick}", fontsize=9)
        page.draw_line(
            fitz.Point(chart_left, y_px),
            fitz.Point(chart_right, y_px),
            color=(0.8, 0.8, 0.8),
            width=0.5,
        )

    # Axis lines
    page.draw_line(
        fitz.Point(chart_left, chart_top), fitz.Point(chart_left, chart_bottom), width=1.5
    )
    page.draw_line(
        fitz.Point(chart_left, chart_bottom), fitz.Point(chart_right, chart_bottom), width=1.5
    )

    # 14 same-color bars + 1 highlighted current-month bar (different fill)
    main_fill = (0.533, 0.463, 0.859)
    highlight_fill = (0.271, 0.0, 0.647)
    for i, val in enumerate(CONTEXT_VALUES):
        bar_top = chart_bottom - (val / max_val) * chart_h
        bar_x0 = chart_left + i * bar_spacing + (bar_spacing - bar_width) / 2
        bar_x1 = bar_x0 + bar_width
        fill = highlight_fill if i == n - 1 else main_fill
        page.draw_rect(fitz.Rect(bar_x0, bar_top, bar_x1, chart_bottom), fill=fill, color=None)
        page.insert_text(
            fitz.Point(bar_x0 + bar_width / 2 - 8, chart_bottom + 12),
            CONTEXT_MONTHS[i],
            fontsize=8,
        )

    # Unrelated section text ~80 pt below chart_bottom.  With the old
    # unbounded x-label search these spans would pollute axes.x.labels.
    env_y = chart_bottom + 80
    page.insert_text(fitz.Point(chart_left, env_y), "Your Environmental Impact", fontsize=10)
    page.insert_text(
        fitz.Point(chart_left, env_y + 15),
        "230 kWh of renewable energy is equivalent to the CO2",
        fontsize=9,
    )
    page.insert_text(fitz.Point(chart_left + 220, env_y), "Your Rewards", fontsize=10)
    page.insert_text(
        fitz.Point(chart_left, env_y + 30), "BAGS OF TRASH RECYCLED", fontsize=9
    )

    doc.save(str(path))
    doc.close()


def _make_bar_no_axis_pdf(path: Path) -> None:
    """Bar chart whose bars are detectable but whose y-axis prints no numeric
    tick labels and whose bars carry no printed value labels either.

    Mirrors a bill where the usage chart is visually present but has no
    readable scale at all, so no source of real values exists.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    chart_left = 80.0
    chart_right = 540.0
    chart_top = 150.0
    chart_bottom = 550.0
    chart_w = chart_right - chart_left
    chart_h = chart_bottom - chart_top

    max_val = max(BAR_VALUES)
    n = len(BAR_VALUES)
    bar_width = chart_w / (n * 1.5)
    bar_spacing = chart_w / n

    # Axis lines only — no tick marks, no numeric labels, no gridlines.
    page.draw_line(
        fitz.Point(chart_left, chart_top), fitz.Point(chart_left, chart_bottom), width=1.5
    )
    page.draw_line(
        fitz.Point(chart_left, chart_bottom), fitz.Point(chart_right, chart_bottom), width=1.5
    )

    bar_fill = (0.22, 0.47, 0.72)
    for i, val in enumerate(BAR_VALUES):
        bar_top = chart_bottom - (val / max_val) * chart_h
        bar_x0 = chart_left + i * bar_spacing + (bar_spacing - bar_width) / 2
        bar_x1 = bar_x0 + bar_width
        page.draw_rect(
            fitz.Rect(bar_x0, bar_top, bar_x1, chart_bottom),
            fill=bar_fill,
            color=None,
        )
        page.insert_text(
            fitz.Point(bar_x0 + bar_width / 2 - 8, chart_bottom + 12),
            MONTHS[i],
            fontsize=8,
            color=(0, 0, 0),
        )

    page.insert_text(fitz.Point(200, 130), "Monthly Usage", fontsize=12)

    doc.save(str(path))
    doc.close()


def _make_raster_pdf(path: Path, source_pdf: Path) -> None:
    """Rasterize the bar chart PDF to create a scanned-image PDF."""
    import fitz

    src_doc = fitz.open(str(source_pdf))
    src_page = src_doc[0]
    w = src_page.rect.width
    h = src_page.rect.height
    pix = src_page.get_pixmap(dpi=150)
    src_doc.close()

    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, pixmap=pix)
    doc.save(str(path))
    doc.close()


def generate_expected(
    name: str, chart_type: str, series_data: list[dict], tolerance_pct: float = 2.0
) -> None:
    expected = {
        "_note": "Values are synthetically generated with known ground truth. Human verification recommended before use with real bills.",
        "chart_type": chart_type,
        "tolerance_pct": tolerance_pct,
        "series": series_data,
    }
    out_path = EXPECTED_DIR / f"{name}.json"
    out_path.write_text(json.dumps(expected, indent=2))
    print(f"  Written: {out_path}")


def main() -> None:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic bar chart PDF...")
    _make_bar_pdf(PDFS_DIR / "bar.pdf")
    generate_expected(
        "bar",
        "bar",
        [
            {
                "type": "bar",
                "unit": "dollars",
                "points": [{"x_label": m, "value": v} for m, v in zip(MONTHS, BAR_VALUES)],
            }
        ],
    )

    print("Generating synthetic line chart PDF...")
    _make_line_pdf(PDFS_DIR / "line.pdf")
    generate_expected(
        "line",
        "line",
        [
            {
                "type": "line",
                "unit": "kwh",
                "points": [{"x_label": m, "value": v} for m, v in zip(MONTHS, LINE_VALUES)],
            }
        ],
    )

    print("Generating synthetic hybrid chart PDF...")
    _make_hybrid_pdf(PDFS_DIR / "hybrid.pdf")
    generate_expected(
        "hybrid",
        "hybrid",
        [
            {
                "type": "bar",
                "unit": "dollars",
                "points": [{"x_label": m, "value": v} for m, v in zip(MONTHS, HYBRID_BAR_VALUES)],
            },
            {
                "type": "line",
                "unit": "kwh",
                "points": [{"x_label": m, "value": v} for m, v in zip(MONTHS, HYBRID_LINE_VALUES)],
            },
        ],
    )

    print("Generating rasterized (scanned) bar chart PDF...")
    _make_raster_pdf(PDFS_DIR / "bar_raster.pdf", PDFS_DIR / "bar.pdf")
    # No expected for raster — pixel-space values are not calibrated without OCR

    print("Generating bar chart PDF with no y-axis scale...")
    _make_bar_no_axis_pdf(PDFS_DIR / "bar_no_axis.pdf")
    # No expected values — the whole point of this fixture is that no scale
    # exists, so values must come back uncalibrated (null).

    print("Generating bar-with-context chart PDF (regression fixture)...")
    _make_bar_with_context_pdf(PDFS_DIR / "bar_with_context.pdf")
    generate_expected(
        "bar_with_context",
        "bar",
        [
            {
                "type": "bar",
                "unit": "dollars",
                "points": [
                    {"x_label": m, "value": v}
                    for m, v in zip(CONTEXT_MONTHS, CONTEXT_VALUES)
                ],
            }
        ],
        tolerance_pct=5.0,
    )

    print(
        "Done. Synthetic fixtures written to "
        "tests/fixtures/pdfs/synthetic/ and tests/fixtures/expected/synthetic/"
    )


if __name__ == "__main__":
    main()
