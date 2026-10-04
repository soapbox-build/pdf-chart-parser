"""The bearer wrapper is the only thing between the public Cloud Run URL and the
upstream parser, which has no auth of its own.

Upstream is stubbed here: CI installs requirements.txt, not the parser's
PyMuPDF/OpenCV stack. The stub's tool sits on a REAL mcp 1.29 FastMCP, the SDK upstream
uses, shaped like upstream's tools. The Dockerfile's build-time checks prove the real
`pdf_chart_parser.server.mcp` still exposes `streamable_http_app()` and comes out strict,
and scripts/refuse-contract probes every real tool.
"""

import asyncio
import importlib
import sys
import types

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

GOOD = "t" * 40
PROBE = "undeclared_argument"


def _install_stub_upstream(monkeypatch):
    async def reached(request):
        return PlainTextResponse("reached upstream")

    real = FastMCP("stub-upstream")

    @real.tool()
    def extract_pdf_document(pdf_url: str | None = None, pages: list[int] | None = None) -> str:
        return "extracted"

    class StubMCP:
        _tool_manager = real._tool_manager

        def call_tool(self, name, arguments):
            return real.call_tool(name, arguments)

        def streamable_http_app(self):
            return Starlette(routes=[Route("/mcp", reached, methods=["GET", "POST"])])

    server = types.ModuleType("pdf_chart_parser.server")
    server.mcp = StubMCP()
    package = types.ModuleType("pdf_chart_parser")
    package.server = server
    monkeypatch.setitem(sys.modules, "pdf_chart_parser", package)
    monkeypatch.setitem(sys.modules, "pdf_chart_parser.server", server)


def _load(monkeypatch, token):
    _install_stub_upstream(monkeypatch)
    if token is None:
        monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MCP_AUTH_TOKEN", token)
    monkeypatch.delitem(sys.modules, "auth_app", raising=False)
    return importlib.import_module("auth_app")


@pytest.fixture
def client(monkeypatch):
    return TestClient(_load(monkeypatch, GOOD).app)


SOURCE = "https://github.com/soapbox-build/pdf-chart-parser"
LINK = f'<{SOURCE}>; rel="source"'


def test_no_authorization_header_is_refused(client):
    resp = client.post("/mcp")
    assert resp.status_code == 401
    assert resp.json() == {"error": "unauthorized", "source": SOURCE}


# AGPL-3.0 section 13: everyone who interacts with the service over a network is offered
# its source, so the offer must reach a caller who is refused as well as one who is let in.


def test_a_refusal_offers_the_source(client):
    assert client.post("/mcp").headers["link"] == LINK


def test_an_accepted_request_offers_the_source(client):
    resp = client.post("/mcp", headers={"Authorization": f"Bearer {GOOD}"})
    assert resp.text == "reached upstream"
    assert resp.headers["link"] == LINK


def test_source_route_answers_without_a_token(client):
    resp = client.get("/source")
    assert resp.status_code == 200
    assert resp.json() == {"source": SOURCE, "license": "AGPL-3.0-or-later"}
    assert resp.headers["link"] == LINK


def test_only_get_source_skips_auth(client):
    assert client.post("/source").status_code == 401
    assert client.get("/mcp").status_code == 401
    assert client.get("/source/").status_code == 401


def test_wrong_token_is_refused(client):
    resp = client.post("/mcp", headers={"Authorization": "Bearer " + "x" * 40})
    assert resp.status_code == 401


def test_right_token_without_bearer_scheme_is_refused(client):
    resp = client.post("/mcp", headers={"Authorization": GOOD})
    assert resp.status_code == 401


def test_right_token_reaches_upstream(client):
    resp = client.post("/mcp", headers={"Authorization": f"Bearer {GOOD}"})
    assert resp.status_code == 200
    assert resp.text == "reached upstream"


def test_short_token_refuses_to_start(monkeypatch):
    with pytest.raises(RuntimeError, match="too short"):
        _load(monkeypatch, "short")


def test_unset_token_refuses_to_start(monkeypatch):
    with pytest.raises(KeyError):
        _load(monkeypatch, None)


# mcp 1.29's FastMCP drops an undeclared argument silently; the wrapper must make
# upstream's tools refuse it by name (strict_tool_arguments.py).


def test_an_undeclared_argument_is_refused_by_name(monkeypatch):
    auth_app = _load(monkeypatch, GOOD)
    with pytest.raises(ToolError) as caught:
        asyncio.run(auth_app.mcp.call_tool("extract_pdf_document", {PROBE: 1}))
    text = str(caught.value).replace(repr({PROBE: 1}), "")
    assert PROBE in text and "Extra inputs are not permitted" in text


def test_declared_arguments_still_reach_the_tool(monkeypatch):
    auth_app = _load(monkeypatch, GOOD)
    out = asyncio.run(auth_app.mcp.call_tool("extract_pdf_document", {"pdf_url": "https://example.com/x.pdf"}))
    assert "extracted" in str(out)
