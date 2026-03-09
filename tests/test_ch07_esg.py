from __future__ import annotations

import json

import pandas as pd

from ifa.esg import (
    IndicatorSpec,
    compute_scores,
    normalize_indicators,
    plot_comparison_panel,
    plot_esg_vs_risk,
    plot_radar,
    set_weights,
)


def _make_raw_esg() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["000001", "000002", "000063", "600519"],
            "year": [2023, 2023, 2023, 2023],
            "carbon_intensity": [95, 70, 120, 40],
            "green_capex_ratio": [0.03, 0.08, 0.02, 0.12],
            "employee_turnover": [0.20, 0.12, 0.18, 0.09],
            "training_coverage": [0.65, 0.80, 0.72, 0.88],
            "independent_director_ratio": [0.33, 0.40, 0.37, 0.45],
            "internal_control_issues": [5, 2, 3, 1],
        }
    )


def _specs() -> list[IndicatorSpec]:
    return [
        IndicatorSpec("carbon_intensity", "E", "negative"),
        IndicatorSpec("green_capex_ratio", "E", "positive"),
        IndicatorSpec("employee_turnover", "S", "negative"),
        IndicatorSpec("training_coverage", "S", "positive"),
        IndicatorSpec("independent_director_ratio", "G", "positive"),
        IndicatorSpec("internal_control_issues", "G", "negative"),
    ]


def test_ch07_score_columns_and_range():
    raw = _make_raw_esg()
    specs = _specs()
    dw, iw = set_weights(specs)
    scored = compute_scores(normalize_indicators(raw, specs), specs, dw, iw)

    required = {"stock_code", "year", "E_score", "S_score", "G_score", "total_score"}
    assert required.issubset(scored.columns)
    val_cols = ["E_score", "S_score", "G_score", "total_score"]
    assert (scored[val_cols] >= 0).all().all()
    assert (scored[val_cols] <= 1).all().all()


def test_ch07_weight_scheme_affects_ranking():
    raw = _make_raw_esg()
    specs = _specs()

    dw_eq, iw = set_weights(specs)
    scored_eq = compute_scores(normalize_indicators(raw, specs), specs, dw_eq, iw)

    dw_gov, _ = set_weights(specs, dim_weights={"E": 0.2, "S": 0.2, "G": 0.6})
    scored_gov = compute_scores(normalize_indicators(raw, specs), specs, dw_gov, iw)

    merged = scored_eq.merge(
        scored_gov,
        on=["stock_code", "year"],
        suffixes=("_eq", "_gov"),
    )
    assert (merged["total_score_eq"] != merged["total_score_gov"]).any()

    # 000001 has weaker governance profile in this synthetic sample.
    row = merged[merged["stock_code"] == "000001"].iloc[0]
    assert row["total_score_gov"] < row["total_score_eq"]


def test_ch07_visual_bridge_and_plots():
    raw = _make_raw_esg()
    specs = _specs()
    dw, iw = set_weights(specs)
    scored = compute_scores(normalize_indicators(raw, specs), specs, dw, iw)

    risk = pd.DataFrame(
        {
            "stock_code": ["000001", "000002", "000063", "600519"],
            "year": [2023, 2023, 2023, 2023],
            "risk_score": [0.62, 0.43, 0.58, 0.28],
        }
    )

    merged, fig_scatter = plot_esg_vs_risk(scored, risk, year=2023)
    assert {"stock_code", "year", "total_score", "risk_score"}.issubset(merged.columns)
    assert len(merged) == 4

    fig_radar = plot_radar(scored, code="000001", year=2023)
    fig_cmp = plot_comparison_panel(scored, year=2023, top_n=3)

    for fig in [fig_scatter, fig_radar, fig_cmp]:
        assert fig is not None
        fig.clf()

    print(json.dumps({"merged_rows": len(merged)}))
