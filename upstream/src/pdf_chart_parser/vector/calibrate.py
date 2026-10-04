"""Calibrate chart axes from text tick labels."""

from __future__ import annotations

import re
from typing import NamedTuple

import fitz
import numpy as np

from pdf_chart_parser.models import Axes, AxisCalibration, AxisCalibrationPoint, AxisInfo
from pdf_chart_parser.vector.text import TextSpan

_MONTH_NAMES = {
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
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
_NUMBER_RE = re.compile(r"^[+\-]?\$?\s*(\d[\d,\.]*)\s*(k|kwh|kw|mwh|therm)?$", re.I)
_UNIT_RE = re.compile(r"kwh|kw|mwh|therm|\$", re.I)

# Maximum horizontal distance a tick label can sit from the plot edge. Generous
# so a value axis with leading empty data columns is still reached; the right
# column is chosen by fit quality, not proximity.
_MAX_AXIS_STRIP = 160.0
# Labels within this horizontal distance (measured at the edge nearest the plot)
# belong to the same axis column.
_AXIS_COLUMN_X_TOL = 12.0
# Minimum vertical span the tick labels must cover to be a real axis.
_MIN_AXIS_Y_SPREAD = 10.0
# Minimum linear fit quality for an accepted value axis.
_MIN_AXIS_R2 = 0.95
# A tick label snaps to a gridline within this many points of its center.
_GRIDLINE_SNAP_TOL = 7.0
# Minimum linear-fit quality required before values derived from an axis are
# trusted as real data rather than calibration noise.
CALIBRATION_R2_THRESHOLD = 0.9


def is_axis_calibrated(axis: AxisCalibration) -> bool:
    """Return True when ``axis`` has enough well-fit tick points to trust it.

    Requires at least two tick points and a linear fit quality above
    ``CALIBRATION_R2_THRESHOLD``; anything less means values computed from
    this axis should not be presented as real data.
    """
    return len(axis.points) >= 2 and axis.r_squared >= CALIBRATION_R2_THRESHOLD


class _FitResult(NamedTuple):
    a: float  # scale (value per pixel)
    b: float  # intercept
    r_squared: float
    points: list[AxisCalibrationPoint]


def calibrate_axes(
    spans: list[TextSpan],
    chart_rect: fitz.Rect,
    value_unit_hint: str = "auto",
    plot_rect: fitz.Rect | None = None,
    bar_rects: list | None = None,
    gridline_ys: list[float] | None = None,
) -> tuple[Axes, list[str]]:
    """Fit y-axis (primary + optional secondary) and x-axis from text spans.

    plot_rect is the tight bars/lines bbox; chart_rect includes label expansion.
    Uses plot_rect for determining which side labels belong to when provided.
    bar_rects, when given, lets x-axis labelling skip the numeric value labels
    printed on the bars of an axis-less chart.

    Returns (Axes, warnings).
    """
    warnings: list[str] = []
    # Use plot_rect for axis-side determination; fall back to chart_rect
    bounds = plot_rect if plot_rect is not None else chart_rect

    y_primary, y_unit, y_warnings = _calibrate_y_axis(
        spans, chart_rect, bounds, side="left", value_unit_hint=value_unit_hint,
        gridline_ys=gridline_ys,
    )
    warnings.extend(y_warnings)
    if y_primary is None:
        warnings.append(
            "y-axis calibration failed: fewer than 2 numeric tick labels found; "
            "values are uncalibrated"
        )

    y_secondary = None
    y_sec_unit = None
    right_fit, r_unit, r_warnings = _calibrate_y_axis(
        spans, chart_rect, bounds, side="right", value_unit_hint="auto",
        gridline_ys=gridline_ys,
    )
    # Require >= 3 points: a 2-point fit is always exact (r² == 1.0), so two
    # stray right-side numbers would otherwise spawn a phantom secondary axis.
    if right_fit is not None and right_fit.r_squared > 0.95 and len(right_fit.points) >= 3:
        y_secondary = right_fit
        y_sec_unit = r_unit
        warnings.append("secondary y-axis detected")

    # When there is no value axis, the numbers printed on the bars are data
    # values, not x-tick labels; keep them out of the x-axis label search.
    exclude_ids: set[int] = set()
    if y_primary is None and bar_rects:
        from pdf_chart_parser.vector.value_labels import read_bar_value_labels

        exclude_ids = {
            id(span) for _, span in read_bar_value_labels(bar_rects, spans).values()
        }

    x_axis, x_labels = _calibrate_x_axis(spans, chart_rect, bounds, exclude_ids=exclude_ids)

    scale_pp = abs(y_primary.a) if y_primary else 0.0
    r2 = y_primary.r_squared if y_primary else 0.0

    if r2 < 0.999 and y_primary and len(y_primary.points) >= 2:
        warnings.append(f"y-axis R² = {r2:.4f} (< 0.999)")

    resolved_unit = _resolve_unit(y_unit, value_unit_hint)

    y_prim_model = AxisCalibration(
        unit=resolved_unit,
        points=y_primary.points if y_primary else [],
        scale_per_point=float(y_primary.a) if y_primary else 0.0,
        intercept=float(y_primary.b) if y_primary else 0.0,
        scale_per_pixel=float(scale_pp),
        r_squared=float(r2),
    )

    y_sec_model = None
    if y_secondary is not None:
        y_sec_model = AxisCalibration(
            unit=_resolve_unit(y_sec_unit, "auto"),
            points=y_secondary.points,
            scale_per_point=float(y_secondary.a),
            intercept=float(y_secondary.b),
            scale_per_pixel=float(abs(y_secondary.a)),
            r_squared=float(y_secondary.r_squared),
        )

    axes = Axes(
        x=x_axis,
        y_primary=y_prim_model,
        y_secondary=y_sec_model,
    )
    return axes, warnings


def _parse_number(text: str) -> float | None:
    t = text.strip().replace(",", "")
    m = _NUMBER_RE.match(t)
    if not m:
        return None
    digits = m.group(1)
    # Reject grouped strings that are not real numbers (e.g. phone numbers like
    # "833.209.5245"), which the digit class would otherwise match.
    if digits.count(".") > 1:
        return None
    try:
        val = float(digits)
    except ValueError:
        return None
    suffix = (m.group(2) or "").lower()
    if suffix == "k":
        val *= 1000
    return val


def _collect_unit(text: str) -> str | None:
    m = _UNIT_RE.search(text)
    if not m:
        return None
    raw = m.group(0).lower()
    if raw == "$":
        return "dollars"
    if raw in ("kwh", "kw", "mwh", "therm"):
        return "kwh"
    return None


def _snap_to_gridline(y: float, gridline_ys: list[float] | None) -> float:
    """Return the gridline y nearest to ``y`` when within tolerance, else ``y``."""
    if not gridline_ys:
        return y
    nearest = min(gridline_ys, key=lambda g: abs(g - y))
    return nearest if abs(nearest - y) <= _GRIDLINE_SNAP_TOL else y


def _r_squared(values: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((values - predicted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0


def _reject_outliers(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop tick points that don't fit the dominant linear trend.

    Correct gridlines can be interleaved with stray numbers that share the axis
    strip. Iteratively remove the worst-fitting point while doing so improves
    the linear fit, keeping at least three points so a genuine 2-point axis is
    never reduced below a fittable set.
    """
    if len(pairs) <= 3:
        return pairs

    current = list(pairs)
    while len(current) > 3:
        ys = np.array([p[1] for p in current])
        values = np.array([p[0] for p in current])
        a, b = np.polyfit(ys, values, 1)
        r2 = _r_squared(values, a * ys + b)
        if r2 >= 0.999:
            break
        # Residual of each point; the largest is the outlier candidate.
        residuals = np.abs(values - (a * ys + b))
        worst = int(np.argmax(residuals))
        trial = current[:worst] + current[worst + 1 :]
        ys_t = np.array([p[1] for p in trial])
        values_t = np.array([p[0] for p in trial])
        at, bt = np.polyfit(ys_t, values_t, 1)
        r2_t = _r_squared(values_t, at * ys_t + bt)
        if r2_t <= r2 + 1e-6:
            break
        current = trial
    return current


def _calibrate_y_axis(
    spans: list[TextSpan],
    chart_rect: fitz.Rect,
    bounds: fitz.Rect,
    side: str,
    value_unit_hint: str,
    gridline_ys: list[float] | None = None,
) -> tuple[_FitResult | None, str | None, list[str]]:
    warnings: list[str] = []
    # Use bounds (plot rect) for side determination, chart_rect for vertical extent
    bx0, by0, bx1, by1 = bounds
    _, cy0, _, cy1 = chart_rect

    # Gather numeric labels on the correct side of the plot. The horizontal reach
    # is generous because the value axis can sit far from the data when the plot
    # has leading empty columns; the right column is then chosen by fit quality,
    # not proximity, so distance alone never selects billing-table numbers.
    if side == "left":
        label_spans = [
            s
            for s in spans
            if s.bbox[2] <= bx0 + 5
            and s.x_center >= bx0 - _MAX_AXIS_STRIP
            and cy0 - 20 <= s.y_center <= cy1 + 20
        ]
    else:
        label_spans = [
            s
            for s in spans
            if s.bbox[0] >= bx1 - 5
            and s.x_center <= bx1 + _MAX_AXIS_STRIP
            and cy0 - 20 <= s.y_center <= cy1 + 20
        ]

    # Cluster on the label edge nearest the plot (right edge for a left axis,
    # left edge for a right axis) so right-aligned numbers like "1000" and "0"
    # stay in one column despite differing centers.
    plot_edge = bx0 if side == "left" else bx1
    triples: list[tuple[float, float, float]] = []  # (value, y_center, edge)
    unit: str | None = None
    for s in label_spans:
        val = _parse_number(s.text)
        if val is not None:
            edge = s.bbox[2] if side == "left" else s.bbox[0]
            # Snap the label's y to the gridline drawn at that value when one sits
            # within half a line-height; the line is the true tick row, the text
            # center only an approximation of it.
            y = _snap_to_gridline(s.y_center, gridline_ys)
            triples.append((val, y, edge))
        u = _collect_unit(s.text)
        if u:
            unit = u

    # Remove calendar-year-like values (1900–2200) when other scale values exist.
    # Year labels on the x-axis can bleed into the search area and corrupt the fit.
    no_years = [t for t in triples if not (1900 <= t[0] <= 2200)]
    if len(no_years) >= 2:
        triples = no_years

    if len(triples) < 2:
        return None, unit, warnings

    pairs = _best_axis_column(triples, plot_edge)
    if pairs is None:
        return None, unit, warnings

    pairs = _reject_outliers(pairs)
    if len(pairs) < 2:
        return None, unit, warnings

    values = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    a, b = (float(v) for v in np.polyfit(ys, values, 1))
    r2 = _r_squared(values, a * ys + b)
    if r2 < _MIN_AXIS_R2:
        return None, unit, warnings

    calib_points = [AxisCalibrationPoint(value=v, y=y) for v, y in pairs]
    result = _FitResult(a=a, b=b, r_squared=r2, points=calib_points)
    return result, unit, warnings


def _best_axis_column(
    triples: list[tuple[float, float, float]],
    plot_edge: float,
) -> list[tuple[float, float]] | None:
    """Pick the vertical column of numbers that best forms a linear value axis.

    Tick labels share an edge position, climb monotonically with y, and span a
    real vertical distance; scattered billing-table numbers do not. Group
    candidates into columns by their edge coordinate, keep those with enough
    vertical spread, and choose the one with the best linear fit (nearest the
    plot breaks ties). Returns the column's (value, y) pairs, or None when no
    column qualifies.
    """
    triples = sorted(triples, key=lambda t: t[2])
    columns: list[list[tuple[float, float, float]]] = [[triples[0]]]
    for t in triples[1:]:
        if t[2] - columns[-1][-1][2] <= _AXIS_COLUMN_X_TOL:
            columns[-1].append(t)
        else:
            columns.append([t])

    best: tuple[float, float] | None = None  # (r2, -distance-to-plot) ranking
    best_pairs: list[tuple[float, float]] | None = None
    for col in columns:
        if len(col) < 2:
            continue
        ys_spread = max(c[1] for c in col) - min(c[1] for c in col)
        if ys_spread < _MIN_AXIS_Y_SPREAD:
            continue
        values = np.array([c[0] for c in col])
        ys = np.array([c[1] for c in col])
        a, b = (float(v) for v in np.polyfit(ys, values, 1))
        r2 = _r_squared(values, a * ys + b)
        mean_edge = sum(c[2] for c in col) / len(col)
        rank = (round(r2, 4), -abs(mean_edge - plot_edge))  # best fit, then nearest plot
        if best is None or rank > best:
            best = rank
            best_pairs = [(c[0], c[1]) for c in col]
    return best_pairs


def y_to_value(y: float, calibration: AxisCalibration) -> float:
    """Convert a y pixel coordinate to a calibrated value.

    Reuses the slope/intercept computed during calibration rather than
    re-fitting on every call.
    """
    if not calibration.points or len(calibration.points) < 2:
        return 0.0
    return calibration.scale_per_point * y + calibration.intercept


def _calibrate_x_axis(
    spans: list[TextSpan],
    chart_rect: fitz.Rect,
    bounds: fitz.Rect | None = None,
    exclude_ids: set[int] | None = None,
) -> tuple[AxisInfo, list[str]]:
    bx0 = bounds.x0 if bounds else chart_rect.x0
    bx1 = bounds.x1 if bounds else chart_rect.x1
    by1 = bounds.y1 if bounds else chart_rect.y1
    plot_w = max(bx1 - bx0, 1.0)
    exclude_ids = exclude_ids or set()

    # Candidate spans: at or below the plot baseline and within the chart rect.
    # The horizontal bound is the (label-expanded) chart rect rather than the
    # tight plot bounds, because outermost x-labels often overhang the data and,
    # for line charts, the label row can be wider than the polyline's bbox.
    cands = [
        s
        for s in spans
        if s.bbox[1] >= by1 - 5
        and s.y_center <= chart_rect.y1 + 10
        and chart_rect.x0 - 5 <= s.x_center <= chart_rect.x1 + 5
        and s.text.strip()
        and id(s) not in exclude_ids
    ]
    if not cands:
        return AxisInfo(kind="categorical", labels=[]), []

    labels, positions = _build_x_labels(cands, plot_w)
    if not labels:
        return AxisInfo(kind="categorical", labels=[]), []

    # Determine categorical vs numeric
    numeric_count = sum(1 for lbl in labels if _parse_number(lbl) is not None)
    kind = "numeric" if numeric_count > len(labels) * 0.7 else "categorical"
    return AxisInfo(kind=kind, labels=labels, positions=positions), []


def _group_rows(spans: list[TextSpan], y_tol: float = 5.0) -> list[list[TextSpan]]:
    """Cluster spans into horizontal rows by y-center proximity."""
    rows: list[list[TextSpan]] = []
    for s in sorted(spans, key=lambda s: s.y_center):
        if rows and s.y_center - rows[-1][-1].y_center <= y_tol:
            rows[-1].append(s)
        else:
            rows.append([s])
    return rows


def _build_x_labels(cands: list[TextSpan], plot_w: float) -> tuple[list[str], list[float]]:
    """Derive ordered x-axis labels from candidate spans below the baseline.

    The label row is the one nearest below the plot baseline that spreads across
    the plot with several entries; its spans define one column each (left to
    right).  Genuine two-line labels (a month row above a year row, or the two
    lines of a "dal …"/"al …" date range) are kept by absorbing a directly
    adjacent lower row only when it has about the same number of entries — one
    per column.  Stray content (temperature readouts, "<"/"=" markers, billing
    tables) differs in count or sits below a larger vertical gap, so it is
    dropped instead of polluting the label list.
    """
    rows = _group_rows(cands)

    def coverage(row: list[TextSpan]) -> float:
        xs = [s.x_center for s in row]
        return (max(xs) - min(xs)) / plot_w

    def qualifies(row: list[TextSpan]) -> bool:
        return len(row) >= 2 and coverage(row) >= 0.3

    # Among the label rows just below the axis, the category row is the richest
    # one — a band of "0" axis end-labels can also span the plot but holds only
    # the two corner values, so entry count, not mere coverage, picks the row.
    qualifying = [i for i, r in enumerate(rows) if qualifies(r)]
    if qualifying:
        top_y = min(min(s.y_center for s in rows[i]) for i in qualifying)
        near = [i for i in qualifying if min(s.y_center for s in rows[i]) - top_y <= 25]
        primary_idx = max(near, key=lambda i: len(rows[i]))
    else:
        primary_idx = max(range(len(rows)), key=lambda i: coverage(rows[i]))

    primary = sorted(rows[primary_idx], key=lambda s: s.x_center)

    # Each primary-row span is a column; lower rows attach to their nearest column.
    columns: list[list[TextSpan]] = [[s] for s in primary]
    column_x = [s.x_center for s in primary]

    for i in range(primary_idx + 1, len(rows)):
        prev_y = max(s.y_center for s in rows[i - 1])
        curr_y = min(s.y_center for s in rows[i])
        if curr_y - prev_y > 12:
            break
        # A true second label line has roughly one entry per column.
        if not (len(primary) * 0.7 <= len(rows[i]) <= len(primary) * 1.3):
            break
        for s in rows[i]:
            nearest = min(range(len(column_x)), key=lambda c: abs(column_x[c] - s.x_center))
            columns[nearest].append(s)

    labels: list[str] = []
    positions: list[float] = []
    for col in columns:
        col.sort(key=lambda s: (s.y_center, s.x_center))
        text = " ".join(s.text.strip() for s in col if s.text.strip())
        if text:
            labels.append(text)
            positions.append(sum(s.x_center for s in col) / len(col))
    return labels, positions


def _resolve_unit(detected: str | None, hint: str) -> str:
    if hint != "auto":
        return hint
    return detected or "auto"
