"""Extract line series data points from stroked polylines."""

from __future__ import annotations

import fitz

from pdf_chart_parser.models import Axes, DataPoint, Series
from pdf_chart_parser.vector.calibrate import is_axis_calibrated, y_to_value
from pdf_chart_parser.vector.color import quantize_color
from pdf_chart_parser.vector.drawings import StrokedPath
from pdf_chart_parser.vector.text import TextSpan, nearest_x_label


def extract_lines(
    line_paths: list[StrokedPath],
    axes: Axes,
    spans: list[TextSpan],
    chart_rect: fitz.Rect,
    plot_rect: fitz.Rect | None = None,
) -> tuple[list[Series], list[str]]:
    """Convert polyline paths to Series data points.

    plot_rect is the tight bars/lines bbox (no label expansion); used for x-mapping.
    Returns ([Series], warnings).
    """
    warnings: list[str] = []
    if not line_paths:
        return [], warnings

    # Use plot_rect for x-label position mapping; fall back to chart_rect
    x_domain = plot_rect if plot_rect is not None else chart_rect

    # Group paths by color
    color_groups: dict[tuple, list[StrokedPath]] = {}
    for path in line_paths:
        key = quantize_color(path.stroke)
        color_groups.setdefault(key, []).append(path)

    # Build legend: small colored text-adjacent swatches → label
    legend = _build_legend(line_paths, spans, chart_rect)

    x_labels = axes.x.labels

    all_series: list[Series] = []
    series_idx = 0

    for color_key, paths in color_groups.items():
        # Merge all points from paths of the same color, sort by x
        all_pts: list[fitz.Point] = []
        for path in paths:
            all_pts.extend(path.points)

        if not all_pts:
            continue

        all_pts.sort(key=lambda p: p.x)

        # Determine which y-axis to use. By the usual dual-axis convention,
        # lines read against the secondary (right) axis when one is present and
        # comparably well-fit. Only keep lines on the primary when both axes
        # share a *known* unit (a same-unit secondary is likely redundant) —
        # the old `unit != unit` test silently failed for unlabeled ("auto")
        # axes and mapped every line to the wrong scale.
        axis_id = "y_primary"
        calibration = axes.y_primary
        if axes.y_secondary is not None:
            sec_r2 = axes.y_secondary.r_squared
            prim_r2 = axes.y_primary.r_squared
            units_known_and_equal = (
                axes.y_secondary.unit != "auto"
                and axes.y_primary.unit != "auto"
                and axes.y_secondary.unit == axes.y_primary.unit
            )
            if sec_r2 >= prim_r2 * 0.98 and not units_known_and_equal:
                axis_id = "y_secondary"
                calibration = axes.y_secondary

        # Lines have no printed-value-label fallback, so a point's value is only
        # trustworthy when the axis it reads against has a usable scale.
        axis_calibrated = is_axis_calibrated(calibration) if calibration is not None else False

        points: list[DataPoint] = []
        for pt in all_pts:
            value = round(y_to_value(pt.y, calibration), 4) if axis_calibrated else None
            x_label = nearest_x_label(pt.x, x_labels, x_domain, axes.x.positions)
            points.append(
                DataPoint(
                    x_label=x_label,
                    x=round(pt.x, 2),
                    value=value,
                    y=round(pt.y, 2),
                    confidence=0.9,
                )
            )

        # Deduplicate points at same x_label, keeping highest confidence
        points = _deduplicate_points(points)

        color_list = list(color_key) if color_key else []
        label = legend.get(color_key, "")
        series = Series(
            id=f"s{series_idx}",
            type="line",
            label=label,
            unit=calibration.unit if calibration else "auto",
            axis=axis_id,
            color=color_list,
            confidence=0.9,
            points=points,
        )
        all_series.append(series)
        series_idx += 1

    if not all_series:
        warnings.append("no line series extracted from paths")

    return all_series, warnings


def _build_legend(
    paths: list[StrokedPath],
    spans: list[TextSpan],
    chart_rect: fitz.Rect,
) -> dict[tuple, str]:
    """Match small colored swatches near text to build a color→label map."""
    legend: dict[tuple, str] = {}
    # Look for small path segments (swatches) near text spans outside the chart plot area
    for path in paths:
        if len(path.points) < 2:
            continue
        bbox = path.bbox
        # Small horizontal segment outside or at the edge of chart — likely a legend swatch
        width = bbox.x1 - bbox.x0
        height = bbox.y1 - bbox.y0
        if width > 30 or height > 10:
            continue
        # Find adjacent text
        for span in spans:
            if abs(span.y_center - (bbox.y0 + bbox.y1) / 2) < 8 and span.bbox[0] > bbox.x1:
                color_key = quantize_color(path.stroke)
                if color_key not in legend:
                    legend[color_key] = span.text.strip()
                    break
    return legend


def _deduplicate_points(points: list[DataPoint]) -> list[DataPoint]:
    """Keep one point per x_label (the one with highest confidence)."""
    seen: dict[str, DataPoint] = {}
    for pt in points:
        key = pt.x_label or str(round(pt.x, 1))
        if key not in seen or pt.confidence > seen[key].confidence:
            seen[key] = pt
    return list(seen.values())
