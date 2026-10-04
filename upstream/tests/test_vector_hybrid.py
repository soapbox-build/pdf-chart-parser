"""Tests for hybrid bar+line chart extraction."""

from __future__ import annotations

from pdf_chart_parser.pipeline import extract_usage_chart


def test_hybrid_chart_found(synthetic_hybrid_pdf):
    result = extract_usage_chart(pdf_path=str(synthetic_hybrid_pdf), return_annotated_image=False)
    assert result["chart_found"]
    assert result["chart_type"] == "hybrid"


def test_hybrid_has_bar_and_line_series(synthetic_hybrid_pdf):
    result = extract_usage_chart(pdf_path=str(synthetic_hybrid_pdf), return_annotated_image=False)
    types = {s["type"] for s in result["series"]}
    assert "bar" in types
    assert "line" in types


def test_hybrid_dual_axes(synthetic_hybrid_pdf):
    result = extract_usage_chart(pdf_path=str(synthetic_hybrid_pdf), return_annotated_image=False)
    assert "y_primary" in result["axes"]
    assert "y_secondary" in result["axes"]


def test_hybrid_values_accurate(synthetic_hybrid_pdf, synthetic_hybrid_expected):
    result = extract_usage_chart(pdf_path=str(synthetic_hybrid_pdf), return_annotated_image=False)
    tolerance = synthetic_hybrid_expected["tolerance_pct"] / 100.0

    for exp_series in synthetic_hybrid_expected["series"]:
        matching = [s for s in result["series"] if s["type"] == exp_series["type"]]
        assert matching, f"No {exp_series['type']} series in result"
        act_series = matching[0]
        exp_map = {p["x_label"]: p["value"] for p in exp_series["points"]}

        n_pass = 0
        n_checked = 0
        for act_pt in act_series["points"]:
            lbl = act_pt.get("x_label", "")
            if lbl in exp_map:
                exp_val = exp_map[lbl]
                if exp_val == 0:
                    continue
                pct_err = abs(act_pt["value"] - exp_val) / abs(exp_val)
                n_checked += 1
                if pct_err <= tolerance:
                    n_pass += 1

        if n_checked > 0:
            assert n_pass >= int(n_checked * 0.9), (
                f"{exp_series['type']}: only {n_pass}/{n_checked} within tolerance"
            )
