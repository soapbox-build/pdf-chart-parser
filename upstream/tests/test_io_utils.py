"""Tests for io_utils.load_pdf_bytes."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest

from pdf_chart_parser.io_utils import load_pdf_bytes

PDFS_DIR = Path(__file__).parent / "fixtures" / "pdfs"


def test_load_from_path(synthetic_bar_pdf):
    data = load_pdf_bytes(pdf_path=str(synthetic_bar_pdf))
    assert data.startswith(b"%PDF")


def test_load_from_base64(synthetic_bar_pdf):
    raw = synthetic_bar_pdf.read_bytes()
    encoded = base64.b64encode(raw).decode()
    data = load_pdf_bytes(pdf_base64=encoded)
    assert data == raw


def test_no_input_raises():
    with pytest.raises(ValueError, match="Exactly one"):
        load_pdf_bytes()


def test_multiple_inputs_raises(synthetic_bar_pdf):
    raw = synthetic_bar_pdf.read_bytes()
    encoded = base64.b64encode(raw).decode()
    with pytest.raises(ValueError, match="exactly one"):
        load_pdf_bytes(pdf_path=str(synthetic_bar_pdf), pdf_base64=encoded)


def test_nonexistent_path_raises():
    with pytest.raises(FileNotFoundError):
        load_pdf_bytes(pdf_path="/no/such/file.pdf")


def test_invalid_base64_raises():
    with pytest.raises(ValueError, match="base64"):
        load_pdf_bytes(pdf_base64="not-valid-base64!!!")


def test_non_pdf_bytes_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(ValueError, match="does not appear to be a PDF"):
        load_pdf_bytes(pdf_path=str(bad))


def test_non_pdf_base64_raises():
    encoded = base64.b64encode(b"not a pdf at all").decode()
    with pytest.raises(ValueError, match="does not appear to be a PDF"):
        load_pdf_bytes(pdf_base64=encoded)


def _make_jpeg_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (200, 150), "white")
    buf = BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def _make_png_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (200, 150), "white")
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_image_base64_is_converted_to_pdf():
    jpeg_bytes = _make_jpeg_bytes()
    encoded = base64.b64encode(jpeg_bytes).decode()
    result = load_pdf_bytes(pdf_base64=encoded)
    assert result.startswith(b"%PDF")


def test_image_path_is_converted_to_pdf(tmp_path):
    png_bytes = _make_png_bytes()
    image_path = tmp_path / "bill.png"
    image_path.write_bytes(png_bytes)
    result = load_pdf_bytes(pdf_path=str(image_path))
    assert result.startswith(b"%PDF")


class _FakeStreamCtx:
    """Mimics the context manager returned by httpx.stream(...)."""

    def __init__(self, response: httpx.Response):
        self._response = response

    def __enter__(self) -> httpx.Response:
        return self._response

    def __exit__(self, *exc_info: object) -> bool:
        return False


def test_load_from_url(monkeypatch, synthetic_bar_pdf):
    raw = synthetic_bar_pdf.read_bytes()
    url = "https://example.com/bill.pdf"

    def fake_stream(method, target_url, **kwargs):
        assert method == "GET"
        assert target_url == url
        req = httpx.Request(method, target_url)
        return _FakeStreamCtx(httpx.Response(200, content=raw, request=req))

    monkeypatch.setattr(httpx, "stream", fake_stream)
    data = load_pdf_bytes(pdf_url=url)
    assert data == raw


def test_image_url_is_converted_to_pdf(monkeypatch):
    jpeg_bytes = _make_jpeg_bytes()
    url = "https://example.com/bill.jpg"

    def fake_stream(method, target_url, **kwargs):
        assert method == "GET"
        assert target_url == url
        req = httpx.Request(method, target_url)
        return _FakeStreamCtx(httpx.Response(200, content=jpeg_bytes, request=req))

    monkeypatch.setattr(httpx, "stream", fake_stream)
    data = load_pdf_bytes(pdf_url=url)
    assert data.startswith(b"%PDF")


def test_pdf_path_containing_url_is_fetched_over_http(monkeypatch, synthetic_bar_pdf):
    # Regression: a presigned download URL forwarded under the wrong
    # argument name (pdf_path instead of pdf_url — e.g. because a caller's
    # own rewrite step swaps a short file reference for a presigned URL
    # without regard to which argument held it) must still be fetched over
    # HTTP rather than treated as a local filesystem path. A URL string can
    # never Path.exists(), so without this fix the call fails instantly
    # with a misleading "PDF not found: <url>" error and never even
    # attempts the download.
    raw = synthetic_bar_pdf.read_bytes()
    url = "https://example.com/bill.pdf?X-Amz-Signature=abc"
    calls: list[str] = []

    def fake_stream(method, target_url, **kwargs):
        calls.append(target_url)
        req = httpx.Request(method, target_url)
        return _FakeStreamCtx(httpx.Response(200, content=raw, request=req))

    monkeypatch.setattr(httpx, "stream", fake_stream)
    data = load_pdf_bytes(pdf_path=url)
    assert data == raw
    assert calls == [url]


def test_pdf_path_url_http_error_is_not_reported_as_not_found(monkeypatch):
    # A real HTTP failure for a URL passed as pdf_path must surface as the
    # normal "HTTP error fetching PDF" message, not the local-file
    # "PDF not found" message — the two failure modes need to stay
    # distinguishable so callers/observability can tell "the object
    # genuinely isn't there" apart from "the download failed".
    url = "https://example.com/missing.pdf"

    def fake_stream(method, target_url, **kwargs):
        req = httpx.Request(method, target_url)
        return _FakeStreamCtx(httpx.Response(404, content=b"", request=req))

    monkeypatch.setattr(httpx, "stream", fake_stream)
    with pytest.raises(ValueError, match="HTTP error fetching PDF"):
        load_pdf_bytes(pdf_path=url)
