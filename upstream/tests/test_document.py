"""Tests for the generic extract_pdf_document tool."""

from __future__ import annotations

import base64
import json
import shutil
from io import BytesIO

import fitz  # pymupdf
import pytest
from PIL import Image, ImageDraw

from pdf_chart_parser import document as doc_mod
from pdf_chart_parser.document import extract_pdf_document


@pytest.fixture
def multipage_pdf_bytes() -> bytes:
    """A small 3-page text PDF for extraction tests."""
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} heading\nSome body text on page {i + 1}.")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def scanned_pdf_bytes() -> bytes:
    """A 1-page PDF with no text layer (simulates a scanned page)."""
    doc = fitz.open()
    page = doc.new_page()
    # Draw a filled rectangle so the page renders to a non-trivial image but
    # carries no extractable text.
    page.draw_rect(fitz.Rect(50, 50, 300, 300), fill=(0.2, 0.4, 0.6))
    data = doc.tobytes()
    doc.close()
    return data


def test_extracts_all_pages_text(multipage_pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(multipage_pdf_bytes).decode("ascii")
    out, images, _native = extract_pdf_document(pdf_base64=b64)
    assert out["total_pages"] == 3
    assert len(out["pages"]) == 3
    assert out["pages"][0]["page"] == 1
    assert "Page 1" in out["pages"][0]["text"]
    # No images requested and pages are text → no image rendered.
    assert out["pages"][0]["image_png_base64"] is None
    assert images == []
    assert out["truncated"] is False


def test_page_selection_is_one_based(multipage_pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(multipage_pdf_bytes).decode("ascii")
    out, _images, _native = extract_pdf_document(pdf_base64=b64, pages=[2])
    assert [p["page"] for p in out["pages"]] == [2]
    assert "Page 2" in out["pages"][0]["text"]


def test_render_page_images_returns_png(multipage_pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(multipage_pdf_bytes).decode("ascii")
    out, images, _native = extract_pdf_document(pdf_base64=b64, render_page_images=True, pages=[1])
    # The old inline field is deprecated and always null now — the PNG bytes
    # live only in the returned image content parts, never duplicated here.
    assert out["pages"][0]["image_png_base64"] is None
    assert len(images) == 1
    img = images[0]
    assert img.type == "image"
    assert img.mimeType == "image/png"
    assert img.meta == {"page": 1}
    # Decodes to a real PNG (magic bytes).
    assert base64.b64decode(img.data).startswith(b"\x89PNG\r\n\x1a\n")


def test_scanned_page_gets_image_even_without_flag(scanned_pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(scanned_pdf_bytes).decode("ascii")
    out, images, _native = extract_pdf_document(pdf_base64=b64, render_page_images=False)
    # Image-only page → PNG returned anyway so a vision model can read it.
    assert out["pages"][0]["image_png_base64"] is None
    assert len(images) == 1
    assert images[0].meta == {"page": 1}
    assert any("scanned" in n or "image-only" in n for n in out["notes"])


def test_image_png_base64_no_longer_duplicates_bytes_in_json(
    multipage_pdf_bytes: bytes,
) -> None:
    """The whole point of this change: the PNG bytes must not exist twice —
    once as a real image content part and once again as a base64 string
    buried inside the JSON-serialized document dict."""
    b64 = base64.b64encode(multipage_pdf_bytes).decode("ascii")
    out, images, _native = extract_pdf_document(pdf_base64=b64, render_page_images=True, pages=[1])
    assert len(images) == 1
    png_b64 = images[0].data
    assert len(png_b64) > 100  # sanity: this is a real, non-trivial PNG payload
    serialized = json.dumps(out)
    assert png_b64 not in serialized
    assert out["pages"][0]["image_png_base64"] is None
    # The document dict still honestly discloses that the field is deprecated
    # and where the bytes actually went.
    assert any("image_png_base64 is deprecated" in n for n in out["notes"])


def test_no_deprecation_note_when_no_images_rendered(multipage_pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(multipage_pdf_bytes).decode("ascii")
    out, images, _native = extract_pdf_document(pdf_base64=b64)
    assert images == []
    assert not any("image_png_base64 is deprecated" in n for n in out["notes"])


def test_to_markdown_called_with_use_ocr_false(
    monkeypatch: pytest.MonkeyPatch, multipage_pdf_bytes: bytes
) -> None:
    """Regression guard for the auto-OCR corruption bug: to_markdown must be
    called with use_ocr=False so pymupdf4llm's own image-heuristic never
    silently discards a page's real, correctly-extracted native text."""
    calls: list[dict] = []
    real_to_markdown = doc_mod.pymupdf4llm.to_markdown

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_to_markdown(*args, **kwargs)

    monkeypatch.setattr(doc_mod.pymupdf4llm, "to_markdown", spy)

    b64 = base64.b64encode(multipage_pdf_bytes).decode("ascii")
    extract_pdf_document(pdf_base64=b64)

    assert len(calls) == 1
    assert calls[0]["use_ocr"] is False


def test_scanned_pdf_still_yields_real_text_with_use_ocr_false(
    synthetic_bar_raster_pdf,
) -> None:
    """Confirms use_ocr=False on the main to_markdown call does not regress
    genuinely scanned/image-only documents: doc_needs_ocr/add_text_layer burns
    a real OCR text layer in *before* to_markdown runs, so to_markdown reads
    that embedded text layer like any other native text — it is not starved
    of the OCR it needs just because the top-level call disables its own
    auto-OCR heuristic."""
    pytest.importorskip(
        "ocrmypdf",
        reason="ocrmypdf not installed; install pdf-chart-parser[ocr] to enable",
    )
    pdf_bytes = synthetic_bar_raster_pdf.read_bytes()
    out, _images, _native = extract_pdf_document(
        pdf_path=None, pdf_base64=base64.b64encode(pdf_bytes).decode("ascii")
    )

    if not any("OCR" in n or "ocrmypdf" in n.lower() for n in out["notes"]):
        pytest.skip("OCRmyPDF could not run in this environment")

    text = out["pages"][0]["text"] or ""
    assert len(text.strip()) >= doc_mod._IMAGE_ONLY_TEXT_THRESHOLD


def test_page_cap_enforced(monkeypatch: pytest.MonkeyPatch, multipage_pdf_bytes: bytes) -> None:
    # Lower the cap to 2 and confirm truncation kicks in on a 3-page doc.
    monkeypatch.setattr(doc_mod, "MAX_PAGES_PROCESSED", 2)
    b64 = base64.b64encode(multipage_pdf_bytes).decode("ascii")
    out, _images, _native = extract_pdf_document(pdf_base64=b64)
    assert len(out["pages"]) == 2
    assert out["truncated"] is True


def test_no_rasterization_after_byte_cap_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the total-image-bytes cap trips, later pages must not even be
    rasterized — get_pixmap must not be called for them at all. Asserting
    only on the returned page list (which already correctly excludes them)
    would miss the actual bug: wasted rasterization work for pages that are
    thrown away."""
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
    data = doc.tobytes()
    doc.close()

    real_get_pixmap = fitz.Page.get_pixmap
    call_count = 0

    def counting_get_pixmap(self, *args, **kwargs):
        nonlocal call_count
        # pymupdf4llm's own to_markdown call also rasterizes internally (with
        # no "dpi" kwarg) for its layout analysis, independent of this
        # function's own page-rendering loop below. document.py's rendering
        # loop always calls get_pixmap(dpi=dpi) explicitly, so filter on that
        # to isolate the calls this test actually cares about.
        if "dpi" in kwargs:
            call_count += 1
        return real_get_pixmap(self, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_pixmap", counting_get_pixmap)

    # First page's PNG alone exceeds this cap, so the cap trips on page 1 and
    # every later page should skip rasterization entirely.
    monkeypatch.setattr(doc_mod, "MAX_TOTAL_IMAGE_BYTES", 100)

    b64 = base64.b64encode(data).decode("ascii")
    out, images, _native = extract_pdf_document(pdf_base64=b64, render_page_images=True)

    assert call_count == 1
    assert out["truncated"] is True
    assert all(p["image_png_base64"] is None for p in out["pages"])
    # Page 1's own PNG already exceeds this artificially tiny cap, so the cap
    # trips immediately and no image survives at all.
    assert images == []


def test_default_image_dpi_is_100(
    monkeypatch: pytest.MonkeyPatch, multipage_pdf_bytes: bytes
) -> None:
    """The default image_dpi (unset) must render at 100 DPI, not the old 150."""
    seen_dpi: list[int] = []
    real_get_pixmap = fitz.Page.get_pixmap

    def spy(self, *args, **kwargs):
        if "dpi" in kwargs:
            seen_dpi.append(kwargs["dpi"])
        return real_get_pixmap(self, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_pixmap", spy)

    b64 = base64.b64encode(multipage_pdf_bytes).decode("ascii")
    extract_pdf_document(pdf_base64=b64, render_page_images=True, pages=[1])

    assert seen_dpi == [100]


@pytest.mark.parametrize(
    ("image_dpi", "expected_dpi"),
    [
        (10, 36),  # below MIN_IMAGE_DPI clamps up to the minimum
        (500, 200),  # above MAX_IMAGE_DPI clamps down to the maximum
        # 50 sits comfortably inside [MIN_IMAGE_DPI=36, MAX_IMAGE_DPI=200] and
        # must pass through unmodified. This is not an arbitrary mid-range
        # value: it's the low-DPI tier a caller uses for a cheap
        # legibility-and-presence glance at manufacturer-cutsheet pages
        # (as opposed to the 100 DPI default used for full-resolution
        # authored-sheet review) — a pinning test that this already-existing
        # clamp accepts it unchanged, since no clamp-boundary test above
        # exercises a value this close to MIN_IMAGE_DPI without itself
        # clamping.
        (50, 50),
    ],
)
def test_image_dpi_clamp_still_holds_at_extremes(
    monkeypatch: pytest.MonkeyPatch,
    multipage_pdf_bytes: bytes,
    image_dpi: int,
    expected_dpi: int,
) -> None:
    seen_dpi: list[int] = []
    real_get_pixmap = fitz.Page.get_pixmap

    def spy(self, *args, **kwargs):
        if "dpi" in kwargs:
            seen_dpi.append(kwargs["dpi"])
        return real_get_pixmap(self, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "get_pixmap", spy)

    b64 = base64.b64encode(multipage_pdf_bytes).decode("ascii")
    extract_pdf_document(
        pdf_base64=b64, render_page_images=True, pages=[1], image_dpi=image_dpi
    )

    assert seen_dpi == [expected_dpi]


def test_images_still_rendered_past_page_30(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for the latent "silently drops images past page 30"
    bug: with MAX_IMAGES_RENDERED raised to 50, a document bigger than the
    old 30-image cap must still get every page rendered, and each page's
    rendered PNG must correlate to the correct page — not be shifted or
    misaligned by the raised cap."""
    total_pages = 38
    doc = fitz.open()
    for i in range(total_pages):
        page = doc.new_page(width=200, height=200)
        # Encode the page index into the red channel so the returned PNG can
        # be checked back against the page it's supposed to belong to.
        red_level = i / (total_pages - 1)
        page.draw_rect(fitz.Rect(0, 0, 200, 200), fill=(red_level, 0.0, 0.0))
        page.insert_text((10, 190), f"Page {i + 1}", color=(1, 1, 1), fontsize=8)
    data = doc.tobytes()
    doc.close()

    b64 = base64.b64encode(data).decode("ascii")
    out, images, _native = extract_pdf_document(pdf_base64=b64, render_page_images=True)

    assert out["total_pages"] == total_pages
    assert len(out["pages"]) == total_pages
    # All pages get an image, well past the old 30-image cap. The inline
    # field is deprecated/always-null now — the image content parts are the
    # only place the rendered PNGs live.
    assert all(p["image_png_base64"] is None for p in out["pages"])
    assert len(images) == total_pages
    assert out["truncated"] is False

    # Spot-check page-to-image correlation for a sample of pages, including
    # several past the old 30-page cap.
    for page_num in (1, 15, 31, 35, 38):
        page_out = out["pages"][page_num - 1]
        assert page_out["page"] == page_num
        img_part = images[page_num - 1]
        assert img_part.meta == {"page": page_num}
        png_bytes = base64.b64decode(img_part.data)
        img = Image.open(BytesIO(png_bytes))
        red, _, _ = img.getpixel((10, 10))[:3]
        expected_red = round(255 * (page_num - 1) / (total_pages - 1))
        assert abs(red - expected_red) <= 3, (
            f"page {page_num}: expected red~={expected_red}, got {red} "
            "(image/page correlation broken)"
        )


def test_mid_document_chunk_reports_absolute_page_numbers() -> None:
    """A chunked caller requesting pages 11..20 out of a >20-page document
    must get back absolute (document-wide) page numbers on both the text
    side (PdfPage.page) and the image side (_meta.page) — 11-20, never
    0-based or chunk-relative 1-10. This is a forward-looking regression
    guard: a later addendum to this same feature relies on absolute page
    numbers surviving a mid-document page-range request."""
    total_pages = 25
    doc = fitz.open()
    for i in range(total_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} heading")
    data = doc.tobytes()
    doc.close()

    b64 = base64.b64encode(data).decode("ascii")
    chunk_pages = list(range(11, 21))  # 1-based, pages 11..20
    out, images, _native = extract_pdf_document(
        pdf_base64=b64, pages=chunk_pages, render_page_images=True
    )

    assert [p["page"] for p in out["pages"]] == chunk_pages
    for i, page_out in enumerate(out["pages"]):
        assert f"Page {chunk_pages[i]} heading" in page_out["text"]

    assert len(images) == len(chunk_pages)
    assert [img.meta["page"] for img in images] == chunk_pages


def test_late_chunk_in_large_document_reports_absolute_pages_and_correct_content() -> None:
    """A LATE, non-evenly-divisible chunk (pages 41-47 of a 47-page document —
    the exact "N=47, final chunk [41,47]" boundary case from the chunked
    two-tier visual-sweep design) must still report absolute page numbers,
    not chunk-relative ones, AND each returned image's actual bytes must
    correspond to the correct page's content — not just the page-number
    field. This extends test_mid_document_chunk_reports_absolute_page_numbers
    (which only exercises a mid-document chunk of a 25-page document) to the
    specific large-document/final-chunk shape a multi-chunk sweep at the
    50-page headroom target actually produces, and adds the
    content-correlation check (via the same per-page red-channel encoding
    used by test_images_still_rendered_past_page_30) that a bare page-number
    assertion alone would not catch (e.g. an off-by-one that renumbers pages
    but happens to shift images in lockstep would still fail this)."""
    total_pages = 47
    doc = fitz.open()
    for i in range(total_pages):
        page = doc.new_page(width=200, height=200)
        # Encode the page index into the red channel so each page's rendered
        # PNG can be checked back against the specific page it should
        # correspond to, independent of the page-number metadata field.
        red_level = i / (total_pages - 1)
        page.draw_rect(fitz.Rect(0, 0, 200, 200), fill=(red_level, 0.0, 0.0))
        page.insert_text((10, 190), f"Page {i + 1} heading", color=(1, 1, 1), fontsize=8)
    data = doc.tobytes()
    doc.close()

    b64 = base64.b64encode(data).decode("ascii")
    # Final chunk of a 10-page chunking scheme over N=47: [10*(5-1)+1, min(10*5, 47)] = [41, 47].
    chunk_pages = list(range(41, 48))  # 1-based, pages 41..47 (7 pages)
    out, images, _native = extract_pdf_document(
        pdf_base64=b64, pages=chunk_pages, render_page_images=True
    )

    assert len(chunk_pages) == 7  # sanity: this is genuinely the ragged final chunk, not 10
    assert [p["page"] for p in out["pages"]] == chunk_pages
    for i, page_out in enumerate(out["pages"]):
        assert f"Page {chunk_pages[i]} heading" in page_out["text"]

    assert len(images) == len(chunk_pages)
    assert [img.meta["page"] for img in images] == chunk_pages

    for i, page_num in enumerate(chunk_pages):
        png_bytes = base64.b64decode(images[i].data)
        img = Image.open(BytesIO(png_bytes))
        red, _, _ = img.getpixel((10, 10))[:3]
        expected_red = round(255 * (page_num - 1) / (total_pages - 1))
        assert abs(red - expected_red) <= 3, (
            f"page {page_num}: expected red~={expected_red}, got {red} "
            "(image/page correlation broken in a late, ragged chunk)"
        )


def test_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        extract_pdf_document()


def test_image_input_is_converted_to_pdf() -> None:
    """A single image (e.g. a photo of a bill) is accepted directly and
    comes back as a one-page document — no separate PDF conversion step
    required by the caller."""
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "USAGE 950 kWh", fill="black")
    buf = BytesIO()
    img.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    out, images, _native = extract_pdf_document(pdf_base64=b64)

    assert out["total_pages"] == 1
    assert len(out["pages"]) == 1
    assert out["pages"][0]["page"] == 1

    if shutil.which("tesseract") is not None:
        text = out["pages"][0]["text"] or ""
        # Either the OCR text layer picked up the string, or a page image
        # came back so a vision model could read it — either is acceptable
        # structural success without pinning exact OCR output.
        assert "950" in text or len(images) > 0
