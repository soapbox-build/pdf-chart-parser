"""Integration tests: full pipeline end-to-end on all fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from pdf_chart_parser.pipeline import extract_usage_chart

PDFS_DIR = Path(__file__).parent / "fixtures" / "pdfs"
EXPECTED_DIR = Path(__file__).parent / "fixtures" / "expected"


def test_bar_chart_integration(synthetic_bar_pdf, synthetic_bar_expected):
    result = extract_usage_chart(pdf_path=str(synthetic_bar_pdf), return_annotated_image=True)
    assert result["chart_found"]
    assert result["method"] == "vector"
    assert result["chart_type"] == "bar"
    assert len(result["series"]) >= 1
    assert result["annotated_png"] is not None
    assert result["annotated_png"][:4] == b"\x89PNG"
    assert result["confidence"] > 0.5
    # Page markdown should be present
    assert isinstance(result.get("page_markdown"), str)


def test_line_chart_integration(synthetic_line_pdf):
    result = extract_usage_chart(pdf_path=str(synthetic_line_pdf), return_annotated_image=False)
    assert result["chart_found"]
    assert result["chart_type"] == "line"
    assert any(s["type"] == "line" for s in result["series"])


def test_hybrid_chart_integration(synthetic_hybrid_pdf):
    result = extract_usage_chart(pdf_path=str(synthetic_hybrid_pdf), return_annotated_image=False)
    assert result["chart_found"]
    assert result["chart_type"] == "hybrid"
    types = {s["type"] for s in result["series"]}
    assert "bar" in types
    assert "line" in types


def test_no_image_flag(synthetic_bar_pdf):
    result = extract_usage_chart(pdf_path=str(synthetic_bar_pdf), return_annotated_image=False)
    assert result.get("annotated_png") is None


def test_bar_values_within_tolerance(synthetic_bar_pdf, synthetic_bar_expected):
    result = extract_usage_chart(pdf_path=str(synthetic_bar_pdf), return_annotated_image=False)
    bar_series = [s for s in result["series"] if s["type"] == "bar"]
    assert bar_series

    exp_pts = synthetic_bar_expected["series"][0]["points"]
    act_pts = bar_series[0]["points"]
    tolerance = synthetic_bar_expected["tolerance_pct"] / 100.0

    n_pass = 0
    for act, exp in zip(act_pts, exp_pts):
        if exp["value"] == 0:
            continue
        pct_err = abs(act["value"] - exp["value"]) / abs(exp["value"])
        if pct_err <= tolerance:
            n_pass += 1

    assert n_pass >= int(len(exp_pts) * 0.9), (
        f"Only {n_pass}/{len(exp_pts)} bars within {tolerance * 100:.1f}% tolerance"
    )


def test_r_squared_acceptable(synthetic_bar_pdf):
    result = extract_usage_chart(pdf_path=str(synthetic_bar_pdf), return_annotated_image=False)
    assert result["axes"]["y_primary"]["r_squared"] >= 0.99


def test_all_fixture_pdfs_run_without_crash():
    """Smoke test: every PDF under fixtures/pdfs (including subdirs) should not raise."""
    for pdf in sorted(PDFS_DIR.rglob("*.pdf")):
        result = extract_usage_chart(pdf_path=str(pdf), return_annotated_image=False)
        assert "chart_found" in result
        assert "method" in result


def test_real_bills_if_present():
    """If real bill PDFs are placed in fixtures/pdfs (top level), test against expected/*.json."""
    real_bills = list(PDFS_DIR.glob("*.pdf"))  # top-level only; synthetic/ is a subdir
    for pdf in real_bills:
        expected_path = EXPECTED_DIR / f"{pdf.stem}.json"
        result = extract_usage_chart(pdf_path=str(pdf), return_annotated_image=False)
        assert result["chart_found"], f"No chart found in real bill: {pdf.name}"

        if not expected_path.exists():
            continue

        expected = json.loads(expected_path.read_text())
        tolerance = expected.get("tolerance_pct", 2.0) / 100.0
        for exp_series in expected.get("series", []):
            matching = [s for s in result["series"] if s["type"] == exp_series["type"]]
            assert matching, f"Missing {exp_series['type']} series for {pdf.name}"
            act_pts = {p["x_label"]: p["value"] for p in matching[0]["points"]}

            n_pass = n_total = 0
            for exp_pt in exp_series["points"]:
                if exp_pt["x_label"] not in act_pts:
                    continue
                exp_val = exp_pt["value"]
                act_val = act_pts[exp_pt["x_label"]]
                if exp_val == 0:
                    continue
                n_total += 1
                if abs(act_val - exp_val) / abs(exp_val) <= tolerance:
                    n_pass += 1

            if n_total > 0:
                assert n_pass >= int(n_total * 0.9), (
                    f"{pdf.name} {exp_series['type']}: {n_pass}/{n_total} within tolerance"
                )
