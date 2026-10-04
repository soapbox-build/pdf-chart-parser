"""Parse page.get_drawings() into normalized rects and stroked paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import fitz


@dataclass
class RectItem:
    rect: fitz.Rect
    fill: tuple[float, ...] | None
    stroke: tuple[float, ...] | None
    width: float
    seqno: int


@dataclass
class StrokedPath:
    points: list[fitz.Point]
    stroke: tuple[float, ...] | None
    width: float
    dashed: bool
    close_path: bool
    seqno: int
    fill: tuple[float, ...] | None = None
    bbox: fitz.Rect = field(default_factory=fitz.Rect)

    def __post_init__(self) -> None:
        if self.points:
            xs = [p.x for p in self.points]
            ys = [p.y for p in self.points]
            self.bbox = fitz.Rect(min(xs), min(ys), max(xs), max(ys))


def collect_drawings(page: fitz.Page) -> dict[str, list]:
    """Return {'rects': [...RectItem], 'paths': [...StrokedPath]}."""
    rects: list[RectItem] = []
    paths: list[StrokedPath] = []

    for seqno, path in enumerate(page.get_drawings()):
        fill = _to_tuple(path.get("fill"))
        stroke = _to_tuple(path.get("color"))
        width = path.get("width") or 0.0
        dashed = bool(path.get("dashes"))
        close_path = bool(path.get("closePath"))
        items: list[Any] = path.get("items", [])

        pts: list[fitz.Point] = []
        has_rect = False

        for item in items:
            kind = item[0]
            if kind == "re":
                r = item[1]
                rects.append(RectItem(rect=r, fill=fill, stroke=stroke, width=width, seqno=seqno))
                has_rect = True
            elif kind == "l":
                pts += [fitz.Point(item[1]), fitz.Point(item[2])]
            elif kind == "c":
                # Bezier: take on-curve anchors (item[1] = start, item[4] = end)
                pts += [fitz.Point(item[1]), fitz.Point(item[4])]
            elif kind == "m":
                pts.append(fitz.Point(item[1]))

        if pts and not has_rect:
            # Deduplicate consecutive identical points
            deduped = [pts[0]]
            for p in pts[1:]:
                if abs(p.x - deduped[-1].x) > 0.01 or abs(p.y - deduped[-1].y) > 0.01:
                    deduped.append(p)

            # If the path has a fill and its points form an axis-aligned rectangle,
            # treat it as a RectItem so bar-group detection can find it.  This handles
            # PDFs that draw bars via four explicit line segments instead of a 're' item.
            if fill is not None:
                rect = _path_to_rect(deduped)
                if rect is None:
                    # Bars are often drawn with rounded corners, giving many edge
                    # points rather than four; accept a filled outline that nearly
                    # fills its bounding box as a rectangle too.
                    rect = _filled_rect_bbox(deduped)
                if rect is not None:
                    rects.append(
                        RectItem(rect=rect, fill=fill, stroke=stroke, width=width, seqno=seqno)
                    )
                    continue

            paths.append(
                StrokedPath(
                    points=deduped,
                    stroke=stroke,
                    width=width,
                    dashed=dashed,
                    close_path=close_path,
                    seqno=seqno,
                    fill=fill,
                )
            )

    return {"rects": rects, "paths": paths}


def _path_to_rect(points: list[fitz.Point]) -> fitz.Rect | None:
    """If the point set forms an axis-aligned rectangle, return the bounding fitz.Rect.

    Accepts a closing duplicate point (e.g. the start point repeated at the end)
    so paths drawn as four 'l' segments pass correctly.
    """
    unique = {(round(p.x, 1), round(p.y, 1)) for p in points}
    if len(unique) != 4:
        return None
    xs = {u[0] for u in unique}
    ys = {u[1] for u in unique}
    if len(xs) != 2 or len(ys) != 2:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        return None
    return fitz.Rect(x0, y0, x1, y1)


def _filled_rect_bbox(points: list[fitz.Point]) -> fitz.Rect | None:
    """Return the bounding box of a filled outline that is essentially a rectangle.

    A rounded-corner bar traces many points around its perimeter rather than the
    four of a sharp rectangle. Such an outline still encloses nearly its whole
    bounding box, whereas triangles, circles, and icons fill far less. Accept the
    shape as a rectangle when its polygon area covers most of the bbox.
    """
    if len(points) < 4:
        return None
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bbox_area = (x1 - x0) * (y1 - y0)
    if bbox_area < 1.0:
        return None
    # Shoelace area of the (closed) outline.
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y - points[j].x * points[i].y
    area = abs(area) / 2.0
    if area / bbox_area < 0.85:
        return None
    return fitz.Rect(x0, y0, x1, y1)


def _to_tuple(color: Any) -> tuple[float, ...] | None:
    if color is None:
        return None
    return tuple(float(c) for c in color)
