"""Extract bar chart data points from detected bar rectangles."""

from __future__ import annotations

import fitz

from pdf_chart_parser.models import Axes, DataPoint, Series
from pdf_chart_parser.vector.calibrate import is_axis_calibrated, y_to_value
from pdf_chart_parser.vector.color import cluster_by_color, quantize_color
from pdf_chart_parser.vector.drawings import RectItem
from pdf_chart_parser.vector.text import TextSpan, nearest_x_label
from pdf_chart_parser.vector.value_labels import read_bar_value_labels


def extract_bars(
    bar_rects: list[RectItem],
    axes: Axes,
    chart_rect: fitz.Rect,
    spans: list[TextSpan] | None = None,
) -> tuple[list[Series], list[str]]:
    """Convert bar rectangles to Series data points.

    Returns ([Series], warnings).
    """
    warnings: list[str] = []
    if not bar_rects:
        return [], warnings

    # When the y-axis carries no usable scale, the chart likely prints each bar's
    # value as text on/near the bar.  Read those labels and use them directly so
    # the series is not reported as all-zeros.
    axis_calibrated = is_axis_calibrated(axes.y_primary)
    text_values: dict[int, float] = {}
    if not axis_calibrated and spans:
        text_values = {
            bid: vs[0] for bid, vs in read_bar_value_labels(bar_rects, spans).items()
        }
        if text_values:
            warnings.append(
                f"y-axis uncalibrated; read {len(text_values)} bar values from printed labels"
            )

    # Sort bars left-to-right
    sorted_bars = sorted(bar_rects, key=lambda r: r.rect.x0)

    # Get x labels
    x_labels = axes.x.labels
    # Map bar x-centers to labels by position; labels span the bars' x extent.
    x_domain = fitz.Rect(
        min(r.rect.x0 for r in sorted_bars), 0, max(r.rect.x1 for r in sorted_bars), 0
    )

    # Group bars by color (each distinct color = one series)
    color_groups = cluster_by_color(sorted_bars, lambda r: r.fill)

    # Merge small color groups (e.g. a single highlighted "current period" bar) into
    # the dominant series so they are not reported as separate one-bar series.
    _MIN_SERIES_BARS = 4
    if len(color_groups) > 1:
        large = [g for g in color_groups if len(g) >= _MIN_SERIES_BARS]
        small = [g for g in color_groups if len(g) < _MIN_SERIES_BARS]
        if large and small:
            dominant = max(large, key=len)
            for g in small:
                dominant.extend(g)
            dominant.sort(key=lambda r: r.rect.x0)
            color_groups = large

    # If multiple colors remain, each is a separate series; otherwise one series
    all_series = []
    series_idx = 0
    for group_bars in color_groups:
        group_bars.sort(key=lambda r: r.rect.x0)
        color_key = quantize_color(group_bars[0].fill)

        # Determine baseline for this color group.
        # For the lower series in a stacked chart all bars share the same y1 (chart
        # baseline).  For the upper (stacked) series y1 varies per bar — each bar sits
        # on top of the corresponding lower bar.  Detect this via the spread of y1
        # values; if spread > 5 pt, use each bar's own y1 as its individual baseline
        # so the extracted value equals just the height of that segment.
        group_bottoms = [r.rect.y1 for r in group_bars]
        median_bottom = sorted(group_bottoms)[len(group_bottoms) // 2]
        bottom_spread = max(group_bottoms) - min(group_bottoms)
        per_bar_baseline = bottom_spread > 5

        if not per_bar_baseline:
            group_baseline_value = y_to_value(median_bottom, axes.y_primary)
            if abs(group_baseline_value) > 5:
                warnings.append(
                    f"non-zero baseline detected: baseline_value={group_baseline_value:.2f}"
                )

        points: list[DataPoint] = []
        for i, bar in enumerate(group_bars):
            x_center = (bar.rect.x0 + bar.rect.x1) / 2

            from_label = id(bar) in text_values
            if from_label:
                # Value read from a printed label rather than measured geometry.
                value = text_values[id(bar)]
                baseline_y = bar.rect.y1
            elif per_bar_baseline:
                # Stacked bar: value = height of just this segment
                bar_baseline_value = y_to_value(bar.rect.y1, axes.y_primary)
                value = y_to_value(bar.rect.y0, axes.y_primary) - bar_baseline_value
                baseline_y = bar.rect.y1
            else:
                value = y_to_value(bar.rect.y0, axes.y_primary) - group_baseline_value
                baseline_y = median_bottom

            if value < -1:
                warnings.append(f"negative bar value {value:.2f} at index {i}; clipping to 0")
                value = 0.0

            # Neither a printed label nor a usable axis scale: the geometry
            # above is not derived from real data (it collapses to 0), so
            # report the value as unknown rather than a fabricated number.
            final_value: float | None = value
            if not from_label and not axis_calibrated:
                final_value = None

            x_label = nearest_x_label(x_center, x_labels, x_domain, axes.x.positions)
            bar_conf = _bar_confidence(bar, group_bars, axes, baseline_y)
            points.append(
                DataPoint(
                    x_label=x_label,
                    x=round(x_center, 2),
                    value=round(final_value, 4) if final_value is not None else None,
                    y=round(bar.rect.y0, 2),
                    baseline_y=round(baseline_y, 2),
                    confidence=bar_conf,
                )
            )

        color_list = list(color_key) if color_key else []
        series = Series(
            id=f"s{series_idx}",
            type="bar",
            label="",
            unit=axes.y_primary.unit,
            axis="y_primary",
            color=color_list,
            confidence=_series_confidence(points),
            points=points,
        )
        all_series.append(series)
        series_idx += 1

    return all_series, warnings


def _bar_confidence(
    bar: RectItem,
    group: list[RectItem],
    axes: Axes,
    baseline_y: float,
) -> float:
    widths = [r.rect.width for r in group]
    median_w = sorted(widths)[len(widths) // 2]
    width_dev = abs(bar.rect.width - median_w) / max(median_w, 1)
    conf = 1.0 - min(width_dev, 0.5)
    if abs(bar.rect.y1 - baseline_y) > 3:
        conf *= 0.9
    return round(conf, 3)


def _series_confidence(points: list[DataPoint]) -> float:
    if not points:
        return 0.0
    return round(sum(p.confidence for p in points) / len(points), 3)
