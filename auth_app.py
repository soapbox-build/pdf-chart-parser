"""Bearer-auth wrapper around the upstream pdf-chart-parser FastMCP app.

Cloud Run deployment requires a static bearer token (MCP_AUTH_TOKEN) on every
request, matching the convention of Soapbox's other MCP servers. The upstream server
has no auth of its own, so this wraps its streamable-http ASGI app with a
middleware that rejects anything without the exact token.

Upstream is AGPL-3.0, so section 13 requires offering this service's source to
everyone who interacts with it over the network. Every response, a refusal
included, carries a `Link: <SOURCE_URL>; rel="source"` header, and GET /source
answers without a token. scripts/pdf-chart-parser-source.py publishes SOURCE_URL.
"""

import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from pdf_chart_parser.server import mcp
from strict_tool_arguments import strict_tool_arguments

SOURCE_URL = "https://github.com/soapbox-build/pdf-chart-parser"
SOURCE_LINK = f'<{SOURCE_URL}>; rel="source"'

TOKEN = os.environ["MCP_AUTH_TOKEN"]
if len(TOKEN) < 32:
    raise RuntimeError("MCP_AUTH_TOKEN is unset or too short")


class BearerAuth(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "GET" and request.url.path == "/source":
            response = JSONResponse({"source": SOURCE_URL, "license": "AGPL-3.0-or-later"})
        else:
            auth = request.headers.get("authorization", "")
            expected = f"Bearer {TOKEN}"
            if not hmac.compare_digest(auth, expected):
                response = JSONResponse({"error": "unauthorized", "source": SOURCE_URL}, status_code=401)
            else:
                response = await call_next(request)
        response.headers["Link"] = SOURCE_LINK
        return response


# Upstream registers its tools on the mcp 1.29 SDK's FastMCP, which silently drops an
# argument no signature declares. Refuse it by name instead (see strict_tool_arguments.py);
# scripts/refuse-contract probes every tool for it.
strict_tool_arguments(mcp)

app = mcp.streamable_http_app()
app.add_middleware(BearerAuth)
