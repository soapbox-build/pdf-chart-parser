FROM python:3.12-slim

# tesseract and the libGL/glib set are what the upstream `raster` extra
# (opencv-python-headless, pytesseract) needs at runtime. git fetches upstream.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tesseract-ocr tesseract-ocr-eng \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.8.13

# Every Python dependency comes from the hashed lock, and nothing else.
COPY requirements.lock /tmp/requirements.lock
RUN uv pip install --system --no-cache-dir --require-hashes -r /tmp/requirements.lock

# The parser itself is upstream, AGPL-3.0, at a PINNED commit. This used to be
# `git clone --depth 1` of the default branch, so every rebuild shipped whatever
# upstream had pushed since, and a failed `[raster]` install fell back silently
# to a build without it. Now the commit is fixed, checked, and installed with
# --no-deps so it cannot pull anything the lock does not name.
ARG UPSTREAM_SHA=a07f3be1791be5211531cf99745f57580c5c080b
RUN git init -q /opt/pdf-chart-parser \
    && git -C /opt/pdf-chart-parser fetch -q --depth 1 \
       https://github.com/haoxinm/pdf-chart-parser.git "$UPSTREAM_SHA" \
    && git -C /opt/pdf-chart-parser checkout -q --detach FETCH_HEAD \
    && test "$(git -C /opt/pdf-chart-parser rev-parse HEAD)" = "$UPSTREAM_SHA" \
    && uv pip install --system --no-cache-dir --no-deps /opt/pdf-chart-parser \
    && rm -rf /opt/pdf-chart-parser

# Fail the build, not the first request, if upstream no longer exposes the
# FastMCP app auth_app.py wraps, or the raster path cannot import.
RUN python -c "from pdf_chart_parser.server import mcp; mcp.streamable_http_app(); import cv2, pytesseract"

WORKDIR /app
COPY auth_app.py /app/auth_app.py

RUN useradd --system --uid 10001 --no-create-home app
USER 10001

ENV MCP_TRANSPORT=streamable-http \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080
CMD ["sh", "-c", "uvicorn auth_app:app --host 0.0.0.0 --port ${PORT:-8080}"]
