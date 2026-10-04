"""CLI entry point for pdf-chart-parser."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from pdf_chart_parser.pipeline import extract_usage_chart
from pdf_chart_parser.server import mcp

app = typer.Typer(help="Extract energy-usage charts from utility-bill PDFs.")


@app.command()
def extract(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write JSON result to file"),
    page: int | None = typer.Option(
        None, "--page", "-p", help="1-based page number hint; omit to auto-detect"
    ),
    chart_type: str = typer.Option("auto", "--chart-type", "-t", help="bar|line|hybrid|auto"),
    value_unit: str = typer.Option("auto", "--unit", "-u", help="dollars|kwh|auto"),
    no_image: bool = typer.Option(False, "--no-image", help="Skip annotated PNG output"),
    dpi: int = typer.Option(200, "--dpi", help="Render DPI for annotated image"),
    image_out: Path | None = typer.Option(None, "--image-out", help="Write annotated PNG to file"),
) -> None:
    """Extract chart data from a PDF and print JSON to stdout."""
    result = extract_usage_chart(
        pdf_path=pdf_path,
        page=page,
        chart_type=chart_type,
        value_unit=value_unit,
        return_annotated_image=not no_image,
        render_dpi=dpi,
    )

    png: bytes | None = result.pop("annotated_png", None)

    if image_out and png:
        image_out.write_bytes(png)
        typer.echo(f"Annotated image written to {image_out}", err=True)

    json_str = json.dumps(result, indent=2)
    if output:
        output.write_text(json_str)
        typer.echo(f"Result written to {output}", err=True)
    else:
        typer.echo(json_str)


@app.command()
def serve(
    transport: str = typer.Option("stdio", "--transport", help="stdio|streamable-http"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Start the MCP server."""
    os.environ["MCP_TRANSPORT"] = transport
    os.environ["HOST"] = host
    os.environ["PORT"] = str(port)
    mcp.run(transport=transport)


if __name__ == "__main__":
    app()
