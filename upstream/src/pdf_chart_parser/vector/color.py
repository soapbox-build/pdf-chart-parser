"""Shared color math and grouping for vector chart detection and extraction.

Detection (locate) and extraction (bars/lines) must agree on what "same color"
means; keeping these helpers in one place avoids the two stages splitting or
merging series differently.
"""

from __future__ import annotations

import math
from collections.abc import Callable

# Color similarity threshold (max Euclidean distance in RGB for "same color").
COLOR_DIST_THRESHOLD = 0.15

# Quantization granularity: channels are snapped to this many buckets.
_QUANTIZE_BUCKETS = 10


def color_distance(a: tuple | None, b: tuple | None) -> float:
    """Euclidean distance between two RGB colors; 1.0 if either is None."""
    if a is None or b is None:
        return 1.0
    n = min(len(a), len(b), 3)
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


def quantize_color(c: tuple | None) -> tuple | None:
    """Snap an RGB color to a coarse grid for use as a grouping key."""
    if c is None:
        return None
    return tuple(round(v * _QUANTIZE_BUCKETS) / _QUANTIZE_BUCKETS for v in c[:3])


def color_saturation(color: tuple | None) -> float:
    if color is None:
        return 0.0
    r, g, b = (color[i] if i < len(color) else 0.0 for i in range(3))
    return max(r, g, b) - min(r, g, b)


def color_lightness(color: tuple | None) -> float:
    if color is None:
        return 0.0
    r, g, b = (color[i] if i < len(color) else 0.0 for i in range(3))
    return (max(r, g, b) + min(r, g, b)) / 2


def cluster_by_color[T](
    items: list[T],
    get_color: Callable[[T], tuple | None],
    threshold: float = COLOR_DIST_THRESHOLD,
) -> list[list[T]]:
    """Greedily group items whose colors are within ``threshold`` of each other.

    Uses the same Euclidean metric as detection so a series identified during
    location is not re-split during extraction.
    """
    groups: list[list[T]] = []
    used: set[int] = set()
    for i, item in enumerate(items):
        if i in used:
            continue
        group = [item]
        used.add(i)
        ci = get_color(item)
        for j in range(i + 1, len(items)):
            if j in used:
                continue
            if color_distance(ci, get_color(items[j])) < threshold:
                group.append(items[j])
                used.add(j)
        groups.append(group)
    return groups
