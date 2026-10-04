"""Extract text spans with bounding boxes from a PDF page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fitz


@dataclass
class TextSpan:
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float = 0.0
    color: int = 0

    @property
    def x_center(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def y_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


def collect_axis_label_rows(spans: list[TextSpan]) -> list[TextSpan]:
    """Return only the topmost horizontal rows from a y-sorted span list.

    Groups spans into rows (same y within 8 pt), then returns rows while the
    gap between consecutive rows is ≤ 10 pt.  This keeps actual x-axis label
    rows (e.g. month abbreviations on one line, year numbers on the next) while
    stopping before billing tables or legend text that sit further below.
    """
    if not spans:
        return []

    rows: list[list[TextSpan]] = [[spans[0]]]
    for s in spans[1:]:
        if s.y_center - rows[-1][-1].y_center < 8:
            rows[-1].append(s)
        else:
            rows.append([s])

    result = list(rows[0])
    for i in range(1, len(rows)):
        prev_max_y = max(s.y_center for s in rows[i - 1])
        curr_min_y = min(s.y_center for s in rows[i])
        if curr_min_y - prev_max_y <= 10:
            result.extend(rows[i])
        else:
            break
    return result


def nearest_x_label(
    x: float,
    labels: list[str],
    x_domain: fitz.Rect,
    positions: list[float] | None = None,
) -> str:
    """Map an x coordinate to the nearest categorical label.

    When `positions` (the x-center of each label) is supplied, the label whose
    position is closest to `x` wins — robust when the data does not span the full
    axis (e.g. a line with leading empty months). Otherwise labels are assumed
    evenly distributed across the plot's x domain. Used by both bar and line
    extraction so they assign labels the same way.
    """
    if not labels:
        return ""
    if positions and len(positions) == len(labels):
        idx = min(range(len(positions)), key=lambda i: abs(positions[i] - x))
        return labels[idx]
    plot_width = x_domain.x1 - x_domain.x0
    if plot_width <= 0:
        return labels[0]
    n = len(labels)
    rel = (x - x_domain.x0) / plot_width
    idx = int(round(rel * (n - 1)))
    idx = max(0, min(idx, n - 1))
    return labels[idx]


def collect_text_spans(page: fitz.Page) -> list[TextSpan]:
    """Return all non-empty text spans from the page with their bboxes."""
    spans: list[TextSpan] = []
    d: dict[str, Any] = page.get_text("dict")
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                bbox = span.get("bbox", (0, 0, 0, 0))
                spans.append(
                    TextSpan(
                        text=text,
                        bbox=tuple(bbox),
                        font_size=span.get("size", 0.0),
                        color=span.get("color", 0),
                    )
                )
    return spans
