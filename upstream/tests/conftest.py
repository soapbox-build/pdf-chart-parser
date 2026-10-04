"""Shared pytest fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SYNTHETIC_PDFS_DIR = FIXTURES_DIR / "pdfs" / "synthetic"
SYNTHETIC_EXPECTED_DIR = FIXTURES_DIR / "expected" / "synthetic"


_REQUIRED_SYNTHETIC_PDFS = [
    "bar.pdf",
    "line.pdf",
    "hybrid.pdf",
    "bar_raster.pdf",
    "bar_with_context.pdf",
    "bar_no_axis.pdf",
]


@pytest.fixture(scope="session", autouse=True)
def ensure_synthetic_fixtures():
    """Generate synthetic test fixtures if any are missing."""
    if not all((SYNTHETIC_PDFS_DIR / p).exists() for p in _REQUIRED_SYNTHETIC_PDFS):
        subprocess.run(
            [sys.executable, str(FIXTURES_DIR / "generate_synthetic.py")],
            check=True,
        )


@pytest.fixture
def synthetic_bar_pdf() -> Path:
    return SYNTHETIC_PDFS_DIR / "bar.pdf"


@pytest.fixture
def synthetic_line_pdf() -> Path:
    return SYNTHETIC_PDFS_DIR / "line.pdf"


@pytest.fixture
def synthetic_hybrid_pdf() -> Path:
    return SYNTHETIC_PDFS_DIR / "hybrid.pdf"


@pytest.fixture
def synthetic_bar_raster_pdf() -> Path:
    return SYNTHETIC_PDFS_DIR / "bar_raster.pdf"


@pytest.fixture
def bar_with_context_pdf() -> Path:
    return SYNTHETIC_PDFS_DIR / "bar_with_context.pdf"


@pytest.fixture
def synthetic_bar_no_axis_pdf() -> Path:
    return SYNTHETIC_PDFS_DIR / "bar_no_axis.pdf"


@pytest.fixture
def bar_with_context_expected() -> dict:
    return json.loads((SYNTHETIC_EXPECTED_DIR / "bar_with_context.json").read_text())


@pytest.fixture
def synthetic_bar_expected() -> dict:
    return json.loads((SYNTHETIC_EXPECTED_DIR / "bar.json").read_text())


@pytest.fixture
def synthetic_line_expected() -> dict:
    return json.loads((SYNTHETIC_EXPECTED_DIR / "line.json").read_text())


@pytest.fixture
def synthetic_hybrid_expected() -> dict:
    return json.loads((SYNTHETIC_EXPECTED_DIR / "hybrid.json").read_text())
