"""Structured application logging: JSON lines to stdout.

This repo has no logging today, so a prod incident's only evidence is
pymupdf4llm's own unstructured stdout chatter, with no way to correlate lines
from a single call or see where time actually went. This module gives every
call a short request-correlation id and a place to emit one structured
summary line per call, read the same way the container's stdout is already
read today (`kubectl logs`).

Only counts, timings, and small enumerated fields belong in these log lines.
Never log a pdf_url (it may be a presigned URL over customer PII-bearing
content) or any page text/image bytes — those are the one thing this module
must never carry.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        return json.dumps(payload, default=str)


class _StdoutHandler(logging.Handler):
    """Writes each record to whatever `sys.stdout` currently is.

    Deliberately does *not* cache `sys.stdout` at construction time the way
    `logging.StreamHandler(sys.stdout)` would: this module's loggers are
    configured once at import time, but the process's stdout stream can be
    swapped afterwards (e.g. test runners that redirect stdout per test), and
    log lines should always follow the live stream rather than a stale
    reference to whatever stdout was at import time.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stdout.write(self.format(record) + "\n")
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


_configured_loggers: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """A stdlib logger that emits one JSON object per line to stdout.

    Idempotent per logger name — safe to call at module import time even if
    the module is imported more than once (e.g. under pytest).
    """
    logger = logging.getLogger(name)
    if name not in _configured_loggers:
        handler = _StdoutHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _configured_loggers.add(name)
    return logger


def new_request_id() -> str:
    """Short random id to correlate every log line for a single tool call."""
    return uuid.uuid4().hex[:12]


def elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since `start` (a time.perf_counter() reading)."""
    return round((time.perf_counter() - start) * 1000, 1)
