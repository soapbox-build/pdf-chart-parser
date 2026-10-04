"""Locate the chart region on the page and classify bar/line/hybrid."""

from __future__ import annotations

import fitz

from pdf_chart_parser.vector.calibrate import _parse_number
from pdf_chart_parser.vector.color import (
    COLOR_DIST_THRESHOLD,
    color_distance,
    color_lightness,
    color_saturation,
    quantize_color,
)
from pdf_chart_parser.vector.drawings import RectItem, StrokedPath
from pdf_chart_parser.vector.text import TextSpan, collect_axis_label_rows

# Minimum bars to consider a valid bar group
MIN_BARS = 4
# Minimum points for a line series to span the plot
MIN_LINE_POINTS = 4
# How far left of the data the crop reaches to include a detached value axis.
_Y_AXIS_LABEL_REACH = 160.0


def locate_chart(
    drawings: dict[str, list],
    spans: list[TextSpan],
    chart_type_hint: str = "auto",
) -> tuple[fitz.Rect, str, list[RectItem], list[StrokedPath], fitz.Rect] | None:
    """Find the chart region, classify its type, and return component geometry.

    Returns (chart_rect, chart_type, bar_rects, line_paths, plot_rect) or None.
    plot_rect is the tight bbox of bars/lines only (no label expansion).
    chart_rect is expanded to include nearby axis labels for rendering.
    """
    rects: list[RectItem] = drawings["rects"]
    paths: list[StrokedPath] = drawings["paths"]

    bar_candidates = _collect_bar_candidates(rects)
    bar_groups = _find_bar_groups(bar_candidates)
    line_series = _find_line_series(paths, spans)

    # Apply hint filtering
    if chart_type_hint == "bar":
        line_series = []
    elif chart_type_hint == "line":
        bar_groups = []

    if not bar_groups and not line_series:
        return None

    # Pick best candidate and collect all bars belonging to that chart
    # (including any upper series in a stacked chart).
    all_chart_bars = _collect_chart_bar_groups(bar_groups, bar_candidates, spans)
    best_lines = _best_line_series(line_series, all_chart_bars)

    if all_chart_bars is None and not best_lines:
        return None

    # Classify
    if all_chart_bars is not None and best_lines:
        detected_type = "hybrid"
    elif all_chart_bars is not None:
        detected_type = "bar"
    else:
        detected_type = "line"

    # Compute tight plot_rect (bars/lines only) and expanded chart_rect (with labels)
    plot_rect = _plot_rect(all_chart_bars or [], best_lines)
    chart_rect = _compute_chart_rect(all_chart_bars or [], best_lines, spans)
    return chart_rect, detected_type, all_chart_bars or [], best_lines, plot_rect


def _collect_bar_candidates(rects: list[RectItem]) -> list[RectItem]:
    """Return all rect candidates that could be chart bars (pre-filter by shape and fill)."""
    return [
        r
        for r in rects
        if r.fill is not None
        and (color_saturation(r.fill) > 0.05 or color_lightness(r.fill) < 0.95)
        and r.rect.height > 5
        and r.rect.width > 2
        and r.rect.height > r.rect.width * 0.5  # taller than wide (bars are vertical)
    ]


def _find_bar_groups(candidates: list[RectItem]) -> list[list[RectItem]]:
    """Group bar candidates into baseline-consistent series groups."""
    if not candidates:
        return []

    # Group by similar fill color and similar width
    groups: list[list[RectItem]] = []
    used = set()

    for i, r in enumerate(candidates):
        if i in used:
            continue
        group = [r]
        used.add(i)
        for j, r2 in enumerate(candidates):
            if j in used:
                continue
            if (
                color_distance(r.fill, r2.fill) < COLOR_DIST_THRESHOLD
                and abs(r.rect.width - r2.rect.width) < r.rect.width * 0.3
            ):
                group.append(r2)
                used.add(j)
        if len(group) >= MIN_BARS:
            groups.append(group)

    # Validate: group members must share a common baseline (y1 cluster)
    valid_groups = []
    for group in groups:
        bottoms = [r.rect.y1 for r in group]
        median_bottom = sorted(bottoms)[len(bottoms) // 2]
        aligned = [r for r in group if abs(r.rect.y1 - median_bottom) < 5]
        if len(aligned) >= MIN_BARS:
            valid_groups.append(aligned)

    # Absorb leftover bars (e.g. a highlighted "current period" bar with a different
    # fill color that forms a group of 1) into any valid group that shares their
    # baseline and whose x-extent is adjacent.
    in_valid: set[int] = {id(r) for grp in valid_groups for r in grp}
    leftovers = [c for c in candidates if id(c) not in in_valid]
    if leftovers and valid_groups:
        absorbed: set[int] = set()
        for group in valid_groups:
            baseline = sorted([r.rect.y1 for r in group])[len(group) // 2]
            grp_x0 = min(r.rect.x0 for r in group)
            grp_x1 = max(r.rect.x1 for r in group)
            avg_w = sum(r.rect.width for r in group) / len(group)
            for c in leftovers:
                if id(c) in absorbed:
                    continue
                cx = (c.rect.x0 + c.rect.x1) / 2
                if (
                    abs(c.rect.y1 - baseline) < 5
                    and grp_x0 - avg_w * 2 <= cx <= grp_x1 + avg_w * 2
                ):
                    group.append(c)
                    absorbed.add(id(c))

    return valid_groups


def _find_stacked_above(
    primary: list[RectItem],
    candidates: list[RectItem],
) -> list[RectItem]:
    """Return bars that sit directly on top of the primary group (stacked chart).

    Searches ALL candidates (not just leftovers) so it can find upper-series bars
    that may already form a partial baseline-consistent group on their own.  The
    caller is responsible for de-duplicating before further processing.

    Matches candidates whose x0 aligns with a primary bar and whose y1 is close
    to that primary bar's y0 (the top of the lower-series bar).
    """
    primary_by_x0: dict[float, RectItem] = {}
    for r in primary:
        key = round(r.rect.x0, 0)
        primary_by_x0[key] = r

    avg_w = sum(r.rect.width for r in primary) / len(primary)
    stacked: list[RectItem] = []
    seen: set[int] = set()

    for c in candidates:
        if id(c) in seen:
            continue
        if abs(c.rect.width - avg_w) > avg_w * 0.5:
            continue
        key = round(c.rect.x0, 0)
        match = primary_by_x0.get(key)
        if match is None:
            for pk, pv in primary_by_x0.items():
                if abs(key - pk) <= 2:
                    match = pv
                    break
        if match is None:
            continue
        if abs(c.rect.y1 - match.rect.y0) < 8:
            stacked.append(c)
            seen.add(id(c))

    return stacked if len(stacked) >= MIN_BARS else []


def _find_line_series(paths: list[StrokedPath], spans: list[TextSpan]) -> list[list[StrokedPath]]:
    """Identify non-axis, data line paths and group by color."""
    if not paths:
        return []

    # Separate out axis/gridline candidates
    data_paths = []
    for path in paths:
        if len(path.points) < MIN_LINE_POINTS:
            continue
        if _is_axis_or_gridline(path):
            continue
        if path.stroke is None:
            continue
        # Must have some saturation (not gray/black axes)
        if color_saturation(path.stroke) < 0.1 and color_lightness(path.stroke) < 0.7:
            continue
        data_paths.append(path)

    if not data_paths:
        return []

    # Group by stroke color
    color_groups: dict[tuple, list[StrokedPath]] = {}
    for path in data_paths:
        key = quantize_color(path.stroke) or (0,)
        color_groups.setdefault(key, []).append(path)

    return [group for group in color_groups.values() if len(group) >= 1]


def _is_axis_or_gridline(path: StrokedPath) -> bool:
    """Return True if the path looks like an axis or gridline (not data)."""
    if len(path.points) < 2:
        return True
    xs = [p.x for p in path.points]
    ys = [p.y for p in path.points]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)

    # Purely horizontal (gridline)
    if y_range < 1.5 and x_range > 20:
        return True
    # Purely vertical (axis line)
    if x_range < 1.5 and y_range > 20:
        return True
    # Very thin gray lines
    if path.stroke is not None:
        sat = color_saturation(path.stroke)
        light = color_lightness(path.stroke)
        if sat < 0.05 and light > 0.5:
            return True

    return False


def _collect_chart_bar_groups(
    groups: list[list[RectItem]],
    candidates: list[RectItem],
    spans: list[TextSpan],
) -> list[RectItem] | None:
    """Return all bars belonging to the primary chart as a deduplicated flat list.

    1. Selects the highest-scoring group as the primary.
    2. Searches all candidates for bars stacked directly on top of primary bars
       (upper series in a stacked chart).
    3. Adds any other validated groups that spatially overlap with the primary
       (e.g. a second series at the same x range).
    4. Deduplicates by bar identity before returning.
    """
    if not groups:
        return None
    primary = _best_bar_group(groups, spans)
    if primary is None:
        return None

    primary_bbox = _rects_bbox(primary)
    seen: set[int] = {id(r) for r in primary}
    all_bars: list[RectItem] = list(primary)

    # Add stacked bars (upper series sitting on top of primary bars)
    stacked = _find_stacked_above(primary, candidates)
    for r in stacked:
        if id(r) not in seen:
            all_bars.append(r)
            seen.add(id(r))

    # Add any other validated groups whose spatial extent overlaps primary
    for group in groups:
        group_bbox = _rects_bbox(group)
        if not _rects_overlap(group_bbox, primary_bbox, margin=5):
            continue
        for r in group:
            if id(r) not in seen:
                all_bars.append(r)
                seen.add(id(r))

    return all_bars


def _best_bar_group(groups: list[list[RectItem]], spans: list[TextSpan]) -> list[RectItem] | None:
    if not groups:
        return None
    if len(groups) == 1:
        return groups[0]

    # Prefer group with more month-like x labels nearby
    best = groups[0]
    best_score = _score_bar_group(groups[0], spans)
    for group in groups[1:]:
        score = _score_bar_group(group, spans)
        if score > best_score:
            best_score = score
            best = group
    return best


def _score_bar_group(group: list[RectItem], spans: list[TextSpan]) -> float:
    bbox = _rects_bbox(group)
    below = [s for s in spans if s.bbox[1] > bbox.y1 and s.bbox[1] < bbox.y1 + 40]
    month_keywords = {
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    }
    month_score = sum(1 for s in below if s.text.strip().lower()[:3] in month_keywords)
    unit_keywords = {"kwh", "$", "dollar", "kw", "usage"}
    bbox_y_center = (bbox.y0 + bbox.y1) / 2
    left = [
        s for s in spans if s.bbox[2] < bbox.x0 and abs(s.y_center - bbox_y_center) < bbox.height
    ]
    unit_score = sum(1 for s in left if any(u in s.text.lower() for u in unit_keywords))
    return month_score + unit_score * 2 + len(group) * 0.1


def _best_line_series(
    series: list[list[StrokedPath]], bar_group: list[RectItem] | None
) -> list[StrokedPath]:
    if not series:
        return []

    # Flatten: if bar group exists, filter lines that spatially overlap with it
    flat: list[StrokedPath] = []
    for group in series:
        for path in group:
            flat.append(path)

    if bar_group is not None:
        bar_bbox = _rects_bbox(bar_group)
        flat = [p for p in flat if _rects_overlap(p.bbox, bar_bbox)]

    return flat


def _rects_bbox(rects: list[RectItem]) -> fitz.Rect:
    x0 = min(r.rect.x0 for r in rects)
    y0 = min(r.rect.y0 for r in rects)
    x1 = max(r.rect.x1 for r in rects)
    y1 = max(r.rect.y1 for r in rects)
    return fitz.Rect(x0, y0, x1, y1)


def _rects_overlap(a: fitz.Rect, b: fitz.Rect, margin: float = 20.0) -> bool:
    return (
        a.x0 < b.x1 + margin
        and a.x1 > b.x0 - margin
        and a.y0 < b.y1 + margin
        and a.y1 > b.y0 - margin
    )


def _plot_rect(bar_rects: list[RectItem], line_paths: list[StrokedPath]) -> fitz.Rect:
    """Tight bbox of only bar/line geometry (no label expansion)."""
    points_x: list[float] = []
    points_y: list[float] = []
    for r in bar_rects:
        points_x += [r.rect.x0, r.rect.x1]
        points_y += [r.rect.y0, r.rect.y1]
    for p in line_paths:
        for pt in p.points:
            points_x.append(pt.x)
            points_y.append(pt.y)
    if not points_x:
        return fitz.Rect(0, 0, 100, 100)
    return fitz.Rect(min(points_x), min(points_y), max(points_x), max(points_y))


def horizontal_gridline_ys(
    paths: list[StrokedPath], plot_rect: fitz.Rect
) -> list[float]:
    """Return y-positions of horizontal gridlines spanning the plot.

    These mark true axis-value rows, letting calibration snap tick labels (whose
    text centers can be a few points off) onto the lines actually drawn at each
    value.
    """
    plot_w = max(plot_rect.x1 - plot_rect.x0, 1.0)
    ys: list[float] = []
    for p in paths:
        if len(p.points) < 2:
            continue
        xs = [q.x for q in p.points]
        pys = [q.y for q in p.points]
        if (
            max(pys) - min(pys) < 1.5
            and (max(xs) - min(xs)) > plot_w * 0.4
            and plot_rect.y0 - 30 <= pys[0] <= plot_rect.y1 + 30
        ):
            ys.append(pys[0])
    return sorted(set(ys))


def _compute_chart_rect(
    bar_rects: list[RectItem],
    line_paths: list[StrokedPath],
    spans: list[TextSpan],
) -> fitz.Rect:
    """Union bbox of bars + lines, expanded to include nearby axis labels."""
    core = _plot_rect(bar_rects, line_paths)

    # Use directional margins: generous left/bottom for axis labels, small top/right
    # to avoid pulling in table headers or other content above/beside the chart.
    # For bar charts the baseline coincides with plot_rect.y1, so 55 pt reaches
    # the x-axis labels.  For line charts the data floats above the axis baseline,
    # so use a much larger bottom margin to capture the labels below the frame.
    bottom_margin = 55 if bar_rects else 250
    search = fitz.Rect(
        core.x0 - 70,           # left: room for y-axis value labels
        core.y0 - 20,           # top: minimal clearance above bars
        core.x1 + 20,           # right: minimal padding
        core.y1 + bottom_margin,
    )
    # Collect all in-range spans, separating those above vs below the data area.
    above_spans: list[TextSpan] = []
    below_spans: list[TextSpan] = []
    for s in spans:
        if not (search.x0 <= s.x_center <= search.x1 and search.y0 <= s.y_center <= search.y1):
            continue
        if s.y_center > core.y1:
            below_spans.append(s)
        else:
            above_spans.append(s)

    below_spans.sort(key=lambda s: s.y_center)

    # For BAR charts: restrict below-bar span expansion to the topmost axis-label
    # rows only.  This prevents billing tables, addresses, and other text just
    # below the x-axis from enlarging the chart_rect and appearing in annotations.
    # For LINE charts the x-axis labels often sit far below the data; skip the
    # row-filter so chart_rect expands to reach them (bottom_margin=250 handles this).
    if bar_rects:
        label_spans = collect_axis_label_rows(below_spans)
    else:
        label_spans = below_spans

    nearby_x: list[float] = []
    nearby_y: list[float] = []
    for s in above_spans + label_spans:
        nearby_x += [s.bbox[0], s.bbox[2]]
        nearby_y += [s.bbox[1], s.bbox[3]]

    # Reach the value axis even when it sits far left of the data (a plot with
    # leading empty columns). Include numeric labels within the plot's vertical
    # band up to the axis-strip distance; the vertical-band gate keeps billing
    # tables below the chart from widening the crop.
    for s in spans:
        if (
            s.bbox[2] <= core.x0 + 5
            and s.x_center >= core.x0 - _Y_AXIS_LABEL_REACH
            and core.y0 - 10 <= s.y_center <= core.y1 + 10
            and _parse_number(s.text) is not None
        ):
            nearby_x += [s.bbox[0], s.bbox[2]]
            nearby_y += [s.bbox[1], s.bbox[3]]

    all_x = [core.x0, core.x1] + nearby_x
    all_y = [core.y0, core.y1] + nearby_y
    # Add a small uniform pad so no axis label is clipped at the boundary.
    _PAD = 5
    return fitz.Rect(
        min(all_x) - _PAD,
        min(all_y) - _PAD,
        max(all_x) + _PAD,
        max(all_y) + _PAD,
    )
