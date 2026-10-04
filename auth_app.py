"""Bearer-auth wrapper around the upstream pdf-chart-parser FastMCP app.

Cloud Run deployment requires a static bearer token (MCP_AUTH_TOKEN) on every
request, matching the soapbox-tools MCP fleet convention. The upstream server
has no auth of its own, so this wraps its streamable-http ASGI app with a
middleware that rejects anything without the exact token.
"""

import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from pdf_chart_parser.server import mcp

TOKEN = os.environ["MCP_AUTH_TOKEN"]
if len(TOKEN) < 32:
    raise RuntimeError("MCP_AUTH_TOKEN is unset or too short")


class BearerAuth(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {TOKEN}"
        if not hmac.compare_digest(auth, expected):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


app = mcp.streamable_http_app()
app.add_middleware(BearerAuth)
