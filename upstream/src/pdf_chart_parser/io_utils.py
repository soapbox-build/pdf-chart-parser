"""PDF input handling: path, base64, or URL → bytes.

Images (JPEG/PNG/GIF/BMP/TIFF/WebP) are also accepted at every entry point:
they are transparently wrapped into a one-page PDF before the existing
extraction pipeline runs, so the rest of the codebase only ever sees PDFs.
"""

from __future__ import annotations

import base64
from pathlib import Path

import fitz  # pymupdf, already a core dependency

_PDF_MAGIC = b"%PDF"
_MAX_URL_BYTES = 50 * 1024 * 1024  # 50 MB

_IMAGE_MAGICS: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpg"),  # JPEG
    (b"\x89PNG\r\n\x1a\n", "png"),  # PNG
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),  # GIF
    (b"BM", "bmp"),  # BMP
    (b"II*\x00", "tif"),
    (b"MM\x00*", "tif"),  # TIFF (LE/BE)
)


def _detect_image_filetype(data: bytes) -> str | None:
    for magic, hint in _IMAGE_MAGICS:
        if data.startswith(magic):
            return hint
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _ensure_pdf_bytes(data: bytes, source: str) -> bytes:
    """Return PDF bytes: pass PDFs through unchanged; wrap a supported image
    into a one-page PDF; raise ValueError for anything else (including HEIC,
    which is intentionally unsupported).
    """
    if data.startswith(_PDF_MAGIC):
        return data
    hint = _detect_image_filetype(data)
    if hint is not None:
        try:
            doc = fitz.open(stream=data, filetype=hint)
            pdf_bytes = doc.convert_to_pdf()
            doc.close()
        except Exception as exc:
            raise ValueError(
                f"{source} looked like an image but could not be converted to PDF: {exc}"
            ) from exc
        if not bytes(pdf_bytes).startswith(_PDF_MAGIC):
            raise ValueError(f"{source} image-to-PDF conversion produced invalid output")
        return bytes(pdf_bytes)
    raise ValueError(
        f"{source} does not appear to be a PDF or a supported image "
        f"(missing %PDF magic bytes and no known image signature)"
    )


def _looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _load_from_url(url: str) -> bytes:
    import httpx

    try:
        with httpx.stream("GET", url, timeout=30, follow_redirects=True) as resp:
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_bytes(chunk_size=65536):
                total += len(chunk)
                if total > _MAX_URL_BYTES:
                    raise ValueError(
                        f"PDF at URL exceeds {_MAX_URL_BYTES // (1024 * 1024)} MB size limit"
                    )
                chunks.append(chunk)
            data = b"".join(chunks)
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"HTTP error fetching PDF: {exc}") from exc
    except httpx.RequestError as exc:
        raise ValueError(f"Network error fetching PDF: {exc}") from exc

    data = _ensure_pdf_bytes(data, f"URL '{url}'")
    return data


def load_pdf_bytes(
    pdf_path: str | None = None,
    pdf_base64: str | None = None,
    pdf_url: str | None = None,
) -> bytes:
    """Return raw PDF bytes from exactly one of the three input sources.

    An image (JPEG/PNG/GIF/BMP/TIFF/WebP) may also be supplied at any of the
    three inputs; it is converted into a one-page PDF before being returned.
    HEIC/HEIF is not supported.

    A caller-supplied `pdf_path` that is actually an http(s) URL (e.g. a
    presigned download URL forwarded under the wrong argument name) is
    fetched the same way as `pdf_url` rather than being treated as a local
    filesystem path — a URL string can never `Path.exists()`, so without
    this it fails instantly with a misleading "PDF not found" error instead
    of ever attempting the download.
    """
    provided = [x for x in (pdf_path, pdf_base64, pdf_url) if x is not None]
    if len(provided) == 0:
        raise ValueError("Exactly one of pdf_path, pdf_base64, or pdf_url must be provided")
    if len(provided) > 1:
        raise ValueError("Provide exactly one of pdf_path, pdf_base64, or pdf_url — got multiple")

    if pdf_path is not None:
        if _looks_like_url(pdf_path):
            return _load_from_url(pdf_path)
        p = Path(pdf_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        data = p.read_bytes()
        data = _ensure_pdf_bytes(data, f"file '{pdf_path}'")
        return data

    if pdf_base64 is not None:
        try:
            data = base64.b64decode(pdf_base64)
        except Exception as exc:
            raise ValueError(f"Invalid base64 data: {exc}") from exc
        data = _ensure_pdf_bytes(data, "base64 input")
        return data

    # pdf_url
    assert pdf_url is not None
    return _load_from_url(pdf_url)
