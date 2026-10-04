# pdf-chart-parser

An MCP server and Python library that extracts energy-usage charts from utility-bill PDFs. It locates the chart, calibrates the axes from the PDF's text layer, and returns structured time-series data alongside an annotated PNG for visual verification — entirely deterministic, no LLM required.

## Features

- **Bar, line, and hybrid (bar + line, dual y-axis)** chart types
- **Vector-first extraction** via PyMuPDF `get_drawings()` / `get_text("dict")`; OpenCV raster fallback for scanned PDFs
- **Scanned-PDF support** via an OCRmyPDF text-layer step (optional `[ocr]` extra): image-only pages get a searchable text layer so they flow through the same high-accuracy text-layer calibration as digital PDFs
- **Full page text** returned as LLM-friendly Markdown (via `pymupdf4llm`)
- **MCP tool** (`extract_usage_chart`) compatible with Claude and other MCP-aware LLMs
- Supports `stdio` transport (local) and `streamable-http` (containerized deployment)
- Returns structured JSON + annotated PNG; numeric data is always text content

## Installation

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager
- **Tesseract OCR** (required only for the `[raster]` extra): `apt-get install tesseract-ocr` or `brew install tesseract`
- **OCRmyPDF system tools** (required only for the `[ocr]` extra, which adds a searchable text layer to scanned PDFs): `apt-get install ghostscript qpdf unpaper pngquant tesseract-ocr` or `brew install ocrmypdf`

### Quickstart

```bash
# Install (vector path only)
uv sync

# Install with raster fallback
uv sync --extra raster

# Install with the OCR text-layer step for scanned PDFs
uv sync --extra ocr

# Install everything
uv sync --extra raster --extra ocr

# Run the CLI
uv run pdf-chart-parser --help

# Run the MCP server (stdio)
uv run python -m pdf_chart_parser.server
```

## Usage

### Python library

```python
from pdf_chart_parser.pipeline import extract_usage_chart

result = extract_usage_chart(pdf_path="bill.pdf", return_annotated_image=True)
print(result["chart_type"])   # "bar" | "line" | "hybrid"
for series in result["series"]:
    print(series["label"], series["points"])
```

### CLI

```bash
uv run pdf-chart-parser extract bill.pdf --output result.json
```

### MCP server

Add to your MCP config (`~/.claude/claude_desktop_config.json` or similar):

```json
{
  "mcpServers": {
    "pdf-chart-parser": {
      "command": "uv",
      "args": ["run", "python", "-m", "pdf_chart_parser.server"],
      "cwd": "/path/to/pdf-chart-parser"
    }
  }
}
```

The server exposes the `extract_usage_chart` tool. It returns:
1. **Page text** — full page as Markdown
2. **Chart reading** — structured JSON (series, axes, confidence, warnings)
3. **Annotated PNG** — cropped chart with calibrated gridlines and data-point markers

## Docker / ECR deployment

```bash
# Build and run locally
./scripts/run_local_server.sh

# Build and push to ECR (set ECR_REPO first)
export ECR_REPO=<account>.dkr.ecr.<region>.amazonaws.com/pdf-chart-parser
./scripts/build_and_push.sh
```

The container starts the server on `streamable-http` at port 8000. A local PDF directory can be bind-mounted to `/data` for ad-hoc testing (see `docker/docker-compose.yml`).

## Manual testing (no LLM)

```bash
# In-process test against fixtures
uv run python scripts/run_manual_tests.py

# Against a running HTTP server
uv run python scripts/run_manual_tests.py --http http://localhost:8000
```

Output is written to `manual_test_output/`.

## License

[AGPL-3.0-or-later](LICENSE). This license is required because the project links against [PyMuPDF](https://pymupdf.readthedocs.io/), which is itself AGPL-3.0 licensed.
