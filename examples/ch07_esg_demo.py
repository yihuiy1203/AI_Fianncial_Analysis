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


def build_demo_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["000001", "000002", "000063", "600519", "300750"],
            "year": [2023, 2023, 2023, 2023, 2023],
            "carbon_intensity": [95, 70, 120, 40, 55],
            "green_capex_ratio": [0.03, 0.08, 0.02, 0.12, 0.10],
            "employee_turnover": [0.20, 0.12, 0.18, 0.09, 0.11],
            "training_coverage": [0.65, 0.80, 0.72, 0.88, 0.84],
            "independent_director_ratio": [0.33, 0.40, 0.37, 0.45, 0.42],
            "internal_control_issues": [5, 2, 3, 1, 2],
        }
    )


def build_specs() -> list[IndicatorSpec]:
    return [
        IndicatorSpec("carbon_intensity", "E", "negative"),
        IndicatorSpec("green_capex_ratio", "E", "positive"),
        IndicatorSpec("employee_turnover", "S", "negative"),
        IndicatorSpec("training_coverage", "S", "positive"),
        IndicatorSpec("independent_director_ratio", "G", "positive"),
        IndicatorSpec("internal_control_issues", "G", "negative"),
    ]


def main() -> None:
    raw = build_demo_raw()
    specs = build_specs()
    dim_weights, ind_weights = set_weights(specs, dim_weights={"E": 0.35, "S": 0.30, "G": 0.35})

    normed = normalize_indicators(raw, specs)
    scores = compute_scores(normed, specs, dim_weights, ind_weights)

    risk = pd.DataFrame(
        {
            "stock_code": ["000001", "000002", "000063", "600519", "300750"],
            "year": [2023, 2023, 2023, 2023, 2023],
            "risk_score": [0.62, 0.43, 0.58, 0.28, 0.31],
        }
    )

    merged, fig_scatter = plot_esg_vs_risk(scores, risk, year=2023)
    fig_radar = plot_radar(scores, code="000001", year=2023)
    fig_cmp = plot_comparison_panel(scores, year=2023, top_n=5)

    for fig in [fig_scatter, fig_radar, fig_cmp]:
        fig.clf()

    print(
        json.dumps(
            {
                "rows": int(len(scores)),
                "top3": scores.nlargest(3, "total_score")[["stock_code", "total_score"]].to_dict(orient="records"),
                "merged_rows": int(len(merged)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
