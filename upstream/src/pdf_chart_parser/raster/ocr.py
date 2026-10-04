"""OCR axis labels from image strips."""

from __future__ import annotations

import re

import numpy as np


def ocr_axis_labels(img_array: np.ndarray) -> list[str]:
    """Extract numeric/text labels from an image strip using pytesseract.

    Returns an empty list gracefully if tesseract is unavailable.
    """
    try:
        import pytesseract
        from PIL import Image

        pil_img = Image.fromarray(img_array)
        text = pytesseract.image_to_string(
            pil_img,
            config="--psm 6 -c tessedit_char_whitelist=0123456789.,kKwWhH$JFMASONDjfmasond",
        )
        tokens = re.split(r"[\s\n]+", text.strip())
        return [t for t in tokens if t]
    except ImportError:
        return []
    except Exception:
        return []


def ocr_axis_values(img_array: np.ndarray) -> list[tuple[float, float]]:
    """Extract (numeric value, y-center pixel) pairs from an axis strip.

    Used to calibrate the raster path's y-axis so bar heights become real
    values instead of raw pixels. Returns an empty list if tesseract is
    unavailable or no numeric labels are recognized.
    """
    try:
        import pytesseract
        from PIL import Image

        pil_img = Image.fromarray(img_array)
        data = pytesseract.image_to_data(
            pil_img,
            config="--psm 6 -c tessedit_char_whitelist=0123456789.,kKwWhH$",
            output_type=pytesseract.Output.DICT,
        )
        pairs: list[tuple[float, float]] = []
        for text, top, height in zip(data["text"], data["top"], data["height"]):
            value = _parse_value(text)
            if value is None:
                continue
            pairs.append((value, float(top) + float(height) / 2.0))
        return pairs
    except ImportError:
        return []
    except Exception:
        return []


def _parse_value(token: str) -> float | None:
    cleaned = token.strip().replace(",", "").replace("$", "")
    m = re.match(r"^(\d+(?:\.\d+)?)", cleaned)
    return float(m.group(1)) if m else None
