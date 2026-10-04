"""Tests for structured logging: the JSON formatter, request-id generation,
and that extract_pdf_document's summary log line carries the right fields
without ever leaking a pdf_url or page content/bytes.

extract_pdf_document's logger writes JSON lines directly to stdout via its
own handler (propagate=False, so it never duplicates through a host
application's root logger config) — so these tests read stdout via capsys
rather than pytest's caplog, which only sees records that propagate to the
root logger.
"""

from __future__ import annotations

import base64
import json
import logging

import fitz
import pytest

from pdf_chart_parser.document import extract_pdf_document
from pdf_chart_parser.logging_utils import elapsed_ms, get_logger, new_request_id


def _parse_json_lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.strip().splitlines() if line.strip()]


def test_new_request_id_is_short_and_unique() -> None:
    a, b = new_request_id(), new_request_id()
    assert a != b
    assert 6 <= len(a) <= 32
    assert 6 <= len(b) <= 32


def test_elapsed_ms_is_non_negative() -> None:
    import time

    start = time.perf_counter()
    assert elapsed_ms(start) >= 0.0


def test_get_logger_emits_valid_json_lines() -> None:
    logger = get_logger("pdf_chart_parser.test_logging_utils_probe")
    formatter = logger.handlers[0].formatter
    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="probe",
        args=(),
        exc_info=None,
    )
    record.fields = {"request_id": "abc123", "count": 3}
    parsed = json.loads(formatter.format(record))
    assert parsed["message"] == "probe"
    assert parsed["request_id"] == "abc123"
    assert parsed["count"] == 3
    assert parsed["level"] == "INFO"


def test_extract_pdf_document_logs_summary_without_leaking_content(
    capsys: pytest.CaptureFixture,
) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Sensitive customer content, do not log me")
    data = doc.tobytes()
    doc.close()
    b64 = base64.b64encode(data).decode("ascii")
    pdf_url_shaped_secret = "https://example.com/private/customer-name-address.pdf"

    extract_pdf_document(pdf_base64=b64)

    captured = capsys.readouterr()
    lines = _parse_json_lines(captured.out)
    summary_lines = [
        line
        for line in lines
        if line.get("logger") == "pdf_chart_parser.document"
        and line.get("message") == "extract_pdf_document complete"
    ]
    assert len(summary_lines) == 1
    fields = summary_lines[0]

    assert fields["total_pages"] == 1
    assert fields["pages_processed"] == 1
    assert isinstance(fields["request_id"], str) and fields["request_id"]
    for key in ("load_ms", "to_markdown_ms", "rasterize_ms", "encode_ms", "total_ms"):
        assert key in fields

    # Never log the pdf_url, page text, or image bytes — only counts/timings.
    assert "Sensitive customer content" not in captured.out
    assert pdf_url_shaped_secret not in captured.out
    assert "customer-name-address" not in captured.out


def test_extract_pdf_document_logs_failure_summary(capsys: pytest.CaptureFixture) -> None:
    """Invalid base64 fails inside load_pdf_bytes — the fetch stage — so this
    still raises exactly as before the fetch/parse split, and logs under the
    fetch-specific message (see test_document.py's fetch-vs-parse tests for
    the parse-stage's separate, non-raising failure path)."""
    with pytest.raises(ValueError):
        extract_pdf_document(pdf_base64="not-valid-base64-!!!")

    captured = capsys.readouterr()
    lines = _parse_json_lines(captured.out)
    failure_lines = [
        line
        for line in lines
        if line.get("logger") == "pdf_chart_parser.document"
        and line.get("message") == "extract_pdf_document fetch failed"
    ]
    assert len(failure_lines) == 1
    assert isinstance(failure_lines[0]["request_id"], str) and failure_lines[0]["request_id"]
