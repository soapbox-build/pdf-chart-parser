"""Pydantic output models for the chart extraction result."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AxisCalibrationPoint(BaseModel):
    value: float
    y: float


class AxisInfo(BaseModel):
    kind: Literal["categorical", "numeric"] = "categorical"
    labels: list[str] = Field(default_factory=list)
    # x-center of each label, left to right; parallel to `labels`. Lets data
    # points map to the nearest label by position rather than even spacing.
    positions: list[float] = Field(default_factory=list)


class AxisCalibration(BaseModel):
    unit: str = "auto"
    points: list[AxisCalibrationPoint] = Field(default_factory=list)
    scale_per_point: float = 0.0
    intercept: float = 0.0
    scale_per_pixel: float = 0.0
    r_squared: float = 0.0


class Axes(BaseModel):
    x: AxisInfo = Field(default_factory=AxisInfo)
    y_primary: AxisCalibration = Field(default_factory=AxisCalibration)
    y_secondary: AxisCalibration | None = None


class DataPoint(BaseModel):
    x_label: str = ""
    x: float = 0.0
    # None when this point's value could not be derived from a trustworthy
    # source (a fitted y-axis scale or a printed value label). `y` and
    # `baseline_y` still carry the detected pixel position/height so the raw
    # geometry remains available even when the value itself is unknown.
    value: float | None = None
    y: float = 0.0
    baseline_y: float | None = None
    confidence: float = 1.0


class Series(BaseModel):
    id: str
    type: Literal["bar", "line"]
    label: str = ""
    unit: str = "auto"
    axis: Literal["y_primary", "y_secondary"] = "y_primary"
    color: list[float] = Field(default_factory=list)
    confidence: float = 1.0
    points: list[DataPoint] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    chart_found: bool = False
    method: Literal["vector", "raster_cv", "failed"] = "failed"
    chart_type: Literal["bar", "line", "hybrid"] | None = None
    page: int = 0
    axes: Axes = Field(default_factory=Axes)
    series: list[Series] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    # Explicit, machine-checkable signal for whether `series[].points[].value`
    # holds real data. False whenever any point's value could not be derived
    # from a usable y-axis scale or a printed value label — callers must not
    # treat such a series as real consumption/cost data.
    values_calibrated: bool = False
    calibration_status: Literal[
        "calibrated", "uncalibrated_axis", "low_confidence", "no_chart"
    ] = "no_chart"
    page_markdown: str = ""
