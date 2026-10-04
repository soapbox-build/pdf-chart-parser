"""Render an annotated PNG of the chart region."""

from __future__ import annotations

import io

import fitz
from PIL import Image, ImageDraw, ImageFont

from pdf_chart_parser.models import Axes, Series

_ANNOTATION_TEXT_COLOR = (30, 30, 30)
_ANNOTATION_CIRCLE_FILL = (255, 255, 255)
_ANNOTATION_CIRCLE_OUTLINE = (40, 40, 40)
_ANNOTATION_TEXT_BG = (245, 245, 245)


def annotate_chart(
    page: fitz.Page,
    chart_rect: fitz.Rect,
    series: list[Series],
    axes: Axes,
    render_dpi: int = 200,
    chart_type: str = "bar",
) -> bytes:
    """Render the chart region with calibrated overlays and return PNG bytes."""
    zoom = render_dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=chart_rect)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)
    _annot_font = ImageFont.load_default(size=13)

    # Convert PDF coordinates to image coordinates
    def pdf_to_img(px: float, py: float) -> tuple[int, int]:
        ix = int((px - chart_rect.x0) * zoom)
        iy = int((py - chart_rect.y0) * zoom)
        return ix, iy

    def pdf_y_to_img_y(py: float) -> int:
        return int((py - chart_rect.y0) * zoom)

    def pdf_x_to_img_x(px: float) -> int:
        return int((px - chart_rect.x0) * zoom)

    # Draw horizontal gridlines for primary y-axis calibration points
    for cp in axes.y_primary.points:
        iy = pdf_y_to_img_y(cp.y)
        draw.line([(0, iy), (img.width, iy)], fill=(0, 180, 0, 200), width=1)
        draw.text((2, iy - 8), f"{cp.value:.0f}", fill=(0, 120, 0))

    # Draw secondary y-axis gridlines if present
    if axes.y_secondary:
        for cp in axes.y_secondary.points:
            iy = pdf_y_to_img_y(cp.y)
            draw.line([(0, iy), (img.width, iy)], fill=(180, 0, 180, 200), width=1)

    # Annotate bar series
    for ser in series:
        if ser.type == "bar":
            for pt in ser.points:
                ix, iy = pdf_to_img(pt.x, pt.y)
                r = 5
                draw.ellipse(
                    [(ix - r, iy - r), (ix + r, iy + r)],
                    fill=_ANNOTATION_CIRCLE_FILL,
                    outline=_ANNOTATION_CIRCLE_OUTLINE,
                    width=2,
                )
                label = f"{pt.value:.1f}" if pt.value is not None else "?"
                tx, ty = ix + 6, iy - 12
                bbox = draw.textbbox((tx, ty), label, font=_annot_font)
                pad = 1
                draw.rectangle(
                    [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                    fill=_ANNOTATION_TEXT_BG,
                )
                draw.text((tx, ty), label, fill=_ANNOTATION_TEXT_COLOR, font=_annot_font)

    # Annotate line series
    for ser in series:
        if ser.type == "line":
            color = _series_color_rgb(ser)
            pts_img = [pdf_to_img(pt.x, pt.y) for pt in ser.points]
            if len(pts_img) >= 2:
                draw.line(pts_img, fill=color, width=2)
            for ix, iy in pts_img:
                r = 3
                draw.ellipse([(ix - r, iy - r), (ix + r, iy + r)], fill=color)

    # Footer
    r2 = axes.y_primary.r_squared
    spp = axes.y_primary.scale_per_pixel
    footer = f"type={chart_type}  unit={axes.y_primary.unit}  scale={spp:.4f}/px  R²={r2:.4f}"
    draw.rectangle([(0, img.height - 16), (img.width, img.height)], fill=(240, 240, 240))
    draw.text((2, img.height - 14), footer, fill=(50, 50, 50))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _series_color_rgb(ser: Series) -> tuple[int, int, int]:
    if ser.color and len(ser.color) >= 3:
        return (
            int(ser.color[0] * 255),
            int(ser.color[1] * 255),
            int(ser.color[2] * 255),
        )
    return (220, 80, 80)
