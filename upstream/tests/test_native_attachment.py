"""Tests for the native-PDF-attachment failsafe on extract_pdf_document.

Covers: page-range vs whole-document attachment, the EmbeddedResource/
BlobResourceContents `_meta` vs `meta` kwarg spelling (same footgun as
ImageContent — verify independently, don't assume it matches), the
fetch-vs-parse failure distinction (a parse failure with bytes already in
hand must still produce a usable native attachment plus an explicit
extraction_failed/reason flag, while a fetch failure has nothing to attach
and still raises), and the pathological-input sanity cap.
"""

from __future__ import annotations

import base64

import fitz  # pymupdf
import pytest
from mcp.types import BlobResourceContents, EmbeddedResource

from pdf_chart_parser import document as doc_mod
from pdf_chart_parser.document import extract_pdf_document


def _make_pdf_bytes(n_pages: int) -> bytes:
    """A synthetic n-page PDF whose page text uniquely identifies each page."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} unique marker {i + 1}")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def ten_page_pdf_bytes() -> bytes:
    return _make_pdf_bytes(10)


def _resource_of(embedded: EmbeddedResource) -> BlobResourceContents:
    assert isinstance(embedded.resource, BlobResourceContents)
    return embedded.resource


def test_attach_native_pages_returns_only_selected_pages(ten_page_pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(ten_page_pdf_bytes).decode("ascii")
    out, images, native = extract_pdf_document(pdf_base64=b64, attach_native_pages=[3, 4])

    assert out["extraction_failed"] is False
    assert len(native) == 1
    embedded = native[0]
    assert isinstance(embedded, EmbeddedResource)
    assert embedded.type == "resource"

    resource = _resource_of(embedded)
    assert resource.mimeType == "application/pdf"
    assert str(resource.uri) == "attachment://planset-native.pdf"

    pdf_bytes = base64.b64decode(resource.blob)
    reopened = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        assert reopened.page_count == 2
        assert "Page 3 unique marker 3" in reopened.load_page(0).get_text()
        assert "Page 4 unique marker 4" in reopened.load_page(1).get_text()
    finally:
        reopened.close()

    assert embedded.meta == {"native_attachment": True, "pages": [3, 4]}


def test_attach_native_document_returns_whole_document(ten_page_pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(ten_page_pdf_bytes).decode("ascii")
    out, _images, native = extract_pdf_document(pdf_base64=b64, attach_native_document=True)

    assert out["extraction_failed"] is False
    assert len(native) == 1
    embedded = native[0]
    resource = _resource_of(embedded)

    pdf_bytes = base64.b64decode(resource.blob)
    assert pdf_bytes == ten_page_pdf_bytes

    reopened = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        assert reopened.page_count == 10
    finally:
        reopened.close()

    assert embedded.meta == {"native_attachment": True, "pages": "all"}


def test_whole_document_wins_when_both_params_set(ten_page_pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(ten_page_pdf_bytes).decode("ascii")
    _out, _images, native = extract_pdf_document(
        pdf_base64=b64, attach_native_document=True, attach_native_pages=[1]
    )
    assert len(native) == 1
    assert native[0].meta == {"native_attachment": True, "pages": "all"}


def test_no_native_attachment_when_not_requested(ten_page_pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(ten_page_pdf_bytes).decode("ascii")
    _out, _images, native = extract_pdf_document(pdf_base64=b64)
    assert native == []


def test_embedded_resource_meta_kwarg_spelling_round_trip() -> None:
    """Pin the exact mcp package behavior for EmbeddedResource/
    BlobResourceContents — verified independently of ImageContent's own
    (already-pinned) behavior, per the plan's explicit warning not to assume
    the two content types share the same pydantic alias config.

    `_meta=` (the wire alias) is the only kwarg spelling that actually
    populates `.meta`; `meta=` silently creates a spurious *extra* field and
    leaves `.meta` as None instead of raising.
    """
    blob = base64.b64encode(b"%PDF-1.4 fake").decode("ascii")

    resource_meta_kwarg = BlobResourceContents(
        uri="attachment://x.pdf", mimeType="application/pdf", blob=blob, meta={"native_attachment": True}
    )
    assert resource_meta_kwarg.meta is None

    resource_underscore_meta_kwarg = BlobResourceContents(
        uri="attachment://x.pdf",
        mimeType="application/pdf",
        blob=blob,
        _meta={"native_attachment": True, "pages": [3, 4]},
    )
    assert resource_underscore_meta_kwarg.meta == {"native_attachment": True, "pages": [3, 4]}

    embedded_meta_kwarg = EmbeddedResource(
        type="resource", resource=resource_underscore_meta_kwarg, meta={"native_attachment": True}
    )
    assert embedded_meta_kwarg.meta is None

    embedded_underscore_meta_kwarg = EmbeddedResource(
        type="resource",
        resource=resource_underscore_meta_kwarg,
        _meta={"native_attachment": True, "pages": [3, 4]},
    )
    assert embedded_underscore_meta_kwarg.meta == {"native_attachment": True, "pages": [3, 4]}

    # Round-trip through JSON (the actual wire path) to confirm the alias
    # survives serialization and re-validation, not just direct attribute
    # access on the in-memory object.
    dumped = embedded_underscore_meta_kwarg.model_dump(mode="json", by_alias=True)
    assert dumped["_meta"] == {"native_attachment": True, "pages": [3, 4]}
    reloaded = EmbeddedResource.model_validate(dumped)
    assert reloaded.meta == {"native_attachment": True, "pages": [3, 4]}


def test_fetch_failure_raises_and_produces_no_attachment() -> None:
    """A fetch failure (bytes never obtained) has nothing to attach — this
    stays a hard failure exactly as before the native-attachment feature
    existed, regardless of whether attach_native_* was requested."""
    with pytest.raises(FileNotFoundError):
        extract_pdf_document(
            pdf_path="/nonexistent/path/does-not-exist.pdf",
            attach_native_document=True,
        )


def test_parse_failure_with_bytes_in_hand_still_attaches_native_pdf(
    monkeypatch: pytest.MonkeyPatch, ten_page_pdf_bytes: bytes
) -> None:
    """A parse-stage failure (fetch already succeeded, pymupdf4llm itself
    then crashes) must not discard the bytes already in hand: the result
    carries extraction_failed=True with a reason, and the requested native
    attachment is still produced from those bytes."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated pymupdf4llm crash")

    monkeypatch.setattr(doc_mod.pymupdf4llm, "to_markdown", boom)

    b64 = base64.b64encode(ten_page_pdf_bytes).decode("ascii")
    out, images, native = extract_pdf_document(
        pdf_base64=b64, attach_native_pages=[3, 4]
    )

    assert out["extraction_failed"] is True
    assert out["reason"] and "simulated pymupdf4llm crash" in out["reason"]
    assert out["pages"] == []
    assert images == []

    assert len(native) == 1
    resource = _resource_of(native[0])
    pdf_bytes = base64.b64decode(resource.blob)
    reopened = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        assert reopened.page_count == 2
        assert "Page 3 unique marker 3" in reopened.load_page(0).get_text()
    finally:
        reopened.close()


def test_parse_failure_without_native_attachment_requested_still_reports_reason(
    monkeypatch: pytest.MonkeyPatch, ten_page_pdf_bytes: bytes
) -> None:
    """Same parse failure, but with no attach_native_* requested: no
    attachment is produced (none was asked for), but the failure is still
    reported explicitly rather than raised, since bytes were obtained."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(doc_mod.pymupdf4llm, "to_markdown", boom)

    b64 = base64.b64encode(ten_page_pdf_bytes).decode("ascii")
    out, images, native = extract_pdf_document(pdf_base64=b64)

    assert out["extraction_failed"] is True
    assert out["reason"] and "simulated crash" in out["reason"]
    assert images == []
    assert native == []


def test_sanity_cap_refuses_oversized_native_attachment(
    monkeypatch: pytest.MonkeyPatch, ten_page_pdf_bytes: bytes
) -> None:
    """The parser applies no provider-specific ceiling of its own — only a
    generous sanity cap against clearly pathological input. Lower the cap
    for a cheap test rather than constructing an actual 200 MB fixture,
    mirroring this test file's MAX_PAGES_PROCESSED-style monkeypatch
    pattern already used in test_document.py."""
    monkeypatch.setattr(doc_mod, "MAX_NATIVE_ATTACHMENT_BYTES", 10)

    b64 = base64.b64encode(ten_page_pdf_bytes).decode("ascii")
    out, _images, native = extract_pdf_document(pdf_base64=b64, attach_native_document=True)

    assert native == []
    assert any("sanity cap" in n for n in out["notes"])


def test_native_pages_out_of_range_produces_no_attachment(ten_page_pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(ten_page_pdf_bytes).decode("ascii")
    out, _images, native = extract_pdf_document(pdf_base64=b64, attach_native_pages=[500])

    assert native == []
    assert any("none of the requested" in n for n in out["notes"])
