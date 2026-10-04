"""Read printed value labels for charts that have no calibrated y-axis.

Some utility bills omit a value axis and instead print each data point's
numeric value as text directly on, above, or below its bar (or beside its line
vertex).  When axis calibration yields no scale, these labels are the only
source of real values, so we associate each numeric span with the data element
it annotates by horizontal alignment and vertical proximity.
"""

from __future__ import annotations

from pdf_chart_parser.vector.calibrate import _parse_number
from pdf_chart_parser.vector.drawings import RectItem
from pdf_chart_parser.vector.text import TextSpan

# A value label must sit within this many points of the bar (above its top or
# below its bottom) to be considered that bar's annotation.
_MAX_VERTICAL_GAP = 18.0


def _numeric_spans(spans: list[TextSpan]) -> list[tuple[TextSpan, float]]:
    out: list[tuple[TextSpan, float]] = []
    for s in spans:
        v = _parse_number(s.text)
        if v is not None:
            out.append((s, v))
    return out


def _median_pitch(centers: list[float]) -> float:
    if len(centers) < 2:
        return 0.0
    gaps = sorted(centers[i + 1] - centers[i] for i in range(len(centers) - 1))
    return gaps[len(gaps) // 2]


def read_bar_value_labels(
    bars: list[RectItem], spans: list[TextSpan]
) -> dict[int, tuple[float, TextSpan]]:
    """Map each bar (by id) to the value printed on/near it, when present.

    A label qualifies when it parses as a number, its x-center aligns with the
    bar (within a tolerance derived from the bar width and column pitch), and it
    sits on, just above, or just below the bar.  The closest such label wins,
    preferring one above the bar top.  Bars with no aligned numeric label are
    omitted from the result.  Returns the value and the span it came from so the
    caller can both report the value and keep that span out of the x-axis labels.
    """
    numeric = _numeric_spans(spans)
    if not numeric:
        return {}

    centers = sorted((b.rect.x0 + b.rect.x1) / 2 for b in bars)
    pitch = _median_pitch(centers)

    result: dict[int, tuple[float, TextSpan]] = {}
    for bar in bars:
        bx = (bar.rect.x0 + bar.rect.x1) / 2
        x_tol = max(bar.rect.width * 0.6, pitch * 0.4, 4.0)
        top, bot = bar.rect.y0, bar.rect.y1

        best_rank: tuple[float, float] | None = None
        best: tuple[float, TextSpan] | None = None
        for s, v in numeric:
            if abs(s.x_center - bx) > x_tol:
                continue
            yc = s.y_center
            if yc < top:
                dist, above = top - yc, 0.0
            elif yc > bot:
                dist, above = yc - bot, 1.0
            else:
                dist, above = 0.0, 0.0
            if dist > _MAX_VERTICAL_GAP:
                continue
            # Prefer the closest label, and a label above the bar over one below.
            rank = (dist, above)
            if best_rank is None or rank < best_rank:
                best_rank, best = rank, (v, s)
        if best is not None:
            result[id(bar)] = best
    return result
