"""The bearer wrapper is the only thing between the public Cloud Run URL and the
upstream parser, which has no auth of its own.

Upstream is stubbed here: CI installs requirements.txt, not the parser's
PyMuPDF/OpenCV stack. The Dockerfile's build-time import check is what proves
the real `pdf_chart_parser.server.mcp` still exposes `streamable_http_app()`.
"""

import importlib
import sys
import types

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

GOOD = "t" * 40


def _install_stub_upstream(monkeypatch):
    async def reached(request):
        return PlainTextResponse("reached upstream")

    class StubMCP:
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
