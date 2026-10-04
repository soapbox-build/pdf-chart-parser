"""MCP server smoke tests — invokes tool function directly without live transport."""

from __future__ import annotations

import base64
import inspect

import fitz  # pymupdf
import pytest


def _make_multipage_pdf_base64(n: int = 3) -> str:
    doc = fitz.open()
    for i in range(n):
        page = doc.new_page()
        # Comfortably over document._IMAGE_ONLY_TEXT_THRESHOLD (16 chars) so
        # these pages are treated as genuine text pages, not scanned/
        # image-only ones that would trigger the always-on image fallback.
        page.insert_text((72, 72), f"Page {i + 1} heading\nSome body text on page {i + 1}.")
    data = doc.tobytes()
    doc.close()
    return base64.b64encode(data).decode("ascii")


def test_extract_pdf_document_wrapper_dpi_default_matches_impl():
    """server.py's @mcp.tool() wrapper declares its own image_dpi default,
    duplicated from document.py's implementation rather than delegating to
    it — so the two can silently drift apart (as happened here: the
    implementation's default moved to 100 while this wrapper stayed at the
    old 150, meaning every real MCP caller that omits image_dpi would still
    have gotten 150). Guard against that drift recurring."""
    from pdf_chart_parser.document import extract_pdf_document as impl
    from pdf_chart_parser.server import extract_pdf_document as wrapped

    # FastMCP's @mcp.tool() decorator returns a FunctionTool wrapping the
    # original function; the original is preserved as .fn for inspection.
    tool_fn = getattr(wrapped, "fn", wrapped)
    wrapper_default = inspect.signature(tool_fn).parameters["image_dpi"].default
    impl_default = inspect.signature(impl).parameters["image_dpi"].default
    assert wrapper_default == impl_default == 100


def test_server_importable():
    from pdf_chart_parser.server import extract_usage_chart, mcp

    assert mcp is not None
    assert callable(extract_usage_chart)


@pytest.mark.parametrize("tool_name", ["extract_usage_chart", "extract_pdf_document"])
def test_tool_annotations_declare_safe_read_only_capability(tool_name):
    """Both tools are read-only, non-mutating, and deterministic, and both can
    fetch a caller-supplied pdf_url over the network — the registered
    ToolAnnotations must say so, so MCP clients can make the same safety
    assumptions without inspecting the implementation."""
    from mcp.types import ToolAnnotations

    from pdf_chart_parser.server import mcp

    tool = mcp._tool_manager._tools[tool_name]
    annotations = tool.annotations

    assert annotations is not None
    assert isinstance(annotations, ToolAnnotations)
    assert annotations.readOnlyHint is True
    assert annotations.destructiveHint is False
    assert annotations.idempotentHint is True
    assert annotations.openWorldHint is True


def test_tool_returns_list(synthetic_bar_pdf):
    from pdf_chart_parser.server import extract_usage_chart as tool

    result = tool(pdf_path=str(synthetic_bar_pdf), return_annotated_image=True)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_tool_first_element_is_dict(synthetic_bar_pdf):
    from pdf_chart_parser.server import extract_usage_chart as tool

    result = tool(pdf_path=str(synthetic_bar_pdf), return_annotated_image=False)
    assert isinstance(result[0], dict)
    assert "chart_found" in result[0]
    assert "series" in result[0]


def test_tool_includes_image_content(synthetic_bar_pdf):
    from mcp.server.fastmcp import Image

    from pdf_chart_parser.server import extract_usage_chart as tool

    result = tool(pdf_path=str(synthetic_bar_pdf), return_annotated_image=True)
    assert len(result) == 2
    assert isinstance(result[1], Image)


def test_tool_no_image_when_flag_false(synthetic_bar_pdf):
    from pdf_chart_parser.server import extract_usage_chart as tool

    result = tool(pdf_path=str(synthetic_bar_pdf), return_annotated_image=False)
    assert len(result) == 1


def test_tool_includes_page_markdown(synthetic_bar_pdf):
    from pdf_chart_parser.server import extract_usage_chart as tool

    result = tool(pdf_path=str(synthetic_bar_pdf), return_annotated_image=False)
    data = result[0]
    assert "page_markdown" in data
    assert isinstance(data["page_markdown"], str)


def test_extract_pdf_document_wrapper_returns_list_of_doc_then_images():
    """server.py's extract_pdf_document wrapper must assemble [doc_dict,
    *images] — mirroring the existing extract_usage_chart pattern — rather
    than returning document.py's bare tuple."""
    from mcp.types import ImageContent

    from pdf_chart_parser.server import extract_pdf_document as tool

    b64 = _make_multipage_pdf_base64(3)
    result = tool(pdf_base64=b64, render_page_images=True)

    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert result[0]["total_pages"] == 3
    assert len(result) == 1 + 3  # doc dict + one image per page
    for i, part in enumerate(result[1:]):
        assert isinstance(part, ImageContent)
        assert part.meta == {"page": i + 1}


def test_extract_pdf_document_wrapper_no_images_when_flag_false_and_text_pages():
    from pdf_chart_parser.server import extract_pdf_document as tool

    b64 = _make_multipage_pdf_base64(3)
    result = tool(pdf_base64=b64, render_page_images=False)

    # Plain text pages, no rendering requested → doc dict only, no images.
    assert len(result) == 1
    assert result[0]["pages"][0]["image_png_base64"] is None


@pytest.mark.anyio
async def test_extract_pdf_document_jsonrpc_content_array_shape():
    """A real in-process MCP client/server round trip (not a direct function
    call) must produce a JSON-RPC content array of one text block followed by
    image blocks in page order, each carrying the correct _meta.page — this
    is the actual wire shape a real MCP client (agent-service) receives."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from pdf_chart_parser.server import mcp as server_instance

    b64 = _make_multipage_pdf_base64(3)
    async with create_connected_server_and_client_session(server_instance) as client:
        result = await client.call_tool(
            "extract_pdf_document",
            {"pdf_base64": b64, "render_page_images": True},
        )

    assert not result.isError
    content = result.content
    assert len(content) == 1 + 3

    assert content[0].type == "text"

    image_parts = content[1:]
    for i, part in enumerate(image_parts):
        assert part.type == "image"
        assert part.mimeType == "image/png"
        assert part.meta == {"page": i + 1}
    # Strictly in page order.
    assert [p.meta["page"] for p in image_parts] == [1, 2, 3]


def test_extract_pdf_document_wrapper_assembles_native_attachment_after_images():
    """server.py's wrapper must assemble [doc_dict, *images,
    *native_attachments] — the native-PDF-attachment failsafe part(s) come
    after any rendered page images, matching the plan's stated order."""
    from mcp.types import EmbeddedResource, ImageContent

    from pdf_chart_parser.server import extract_pdf_document as tool

    b64 = _make_multipage_pdf_base64(3)
    result = tool(pdf_base64=b64, render_page_images=True, attach_native_pages=[1])

    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert len(result) == 1 + 3 + 1  # doc dict + 3 images + 1 native attachment
    for part in result[1:4]:
        assert isinstance(part, ImageContent)
    native = result[4]
    assert isinstance(native, EmbeddedResource)
    assert native.meta == {"native_attachment": True, "pages": [1]}


def test_extract_pdf_document_wrapper_no_native_attachment_by_default():
    from pdf_chart_parser.server import extract_pdf_document as tool

    b64 = _make_multipage_pdf_base64(3)
    result = tool(pdf_base64=b64, render_page_images=False)

    assert len(result) == 1  # doc dict only — no attach_native_* requested


@pytest.mark.anyio
async def test_extract_pdf_document_jsonrpc_native_attachment_wire_shape():
    """A real in-process MCP client/server round trip for the
    attach_native_pages failsafe must produce a JSON-RPC content array
    carrying a real `resource` content block with a base64 `blob` and the
    expected `_meta`, not an inert string or a byte-indexed object."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from pdf_chart_parser.server import mcp as server_instance

    b64 = _make_multipage_pdf_base64(5)
    async with create_connected_server_and_client_session(server_instance) as client:
        result = await client.call_tool(
            "extract_pdf_document",
            {"pdf_base64": b64, "attach_native_pages": [2, 3]},
        )

    assert not result.isError
    content = result.content
    assert content[0].type == "text"
    resource_parts = [c for c in content if c.type == "resource"]
    assert len(resource_parts) == 1
    part = resource_parts[0]
    assert part.resource.mimeType == "application/pdf"
    assert part.meta == {"native_attachment": True, "pages": [2, 3]}
    assert isinstance(part.resource.blob, str) and len(part.resource.blob) > 100


@pytest.mark.anyio
async def test_extract_pdf_document_jsonrpc_mid_chunk_absolute_pages():
    """A chunked pages:[11..20]-style call over the wire must report absolute
    page numbers in each image's _meta.page, not 1-based-within-the-chunk —
    a forward-looking regression guard for a later chunked-sweep addendum."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from pdf_chart_parser.server import mcp as server_instance

    b64 = _make_multipage_pdf_base64(25)
    chunk_pages = list(range(11, 21))
    async with create_connected_server_and_client_session(server_instance) as client:
        result = await client.call_tool(
            "extract_pdf_document",
            {"pdf_base64": b64, "pages": chunk_pages, "render_page_images": True},
        )

    assert not result.isError
    image_parts = [c for c in result.content if c.type == "image"]
    assert [p.meta["page"] for p in image_parts] == chunk_pages
