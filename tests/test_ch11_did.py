from __future__ import annotations

import numpy as np
import pandas as pd

from ifa.causal.did import (
    DIDConfig,
    plot_parallel_trends,
    prepare_did_data,
    run_did_regression,
    run_dynamic_did,
    run_placebo_group,
    run_placebo_test,
    summarize_models,
)


def _make_panel(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firms = [f"{i:06d}" for i in range(1, 81)]
    years = list(range(2017, 2025))
    policy_year = 2021

    rows = []
    for i, f in enumerate(firms):
        treat = 1 if i < 40 else 0
        firm_fe = rng.normal(0, 0.8)
        trend_slope = rng.normal(0.02, 0.01)
        for y in years:
            year_fe = 0.1 * (y - years[0])
            post = 1 if y >= policy_year else 0
            did = treat * post
            x_size = rng.normal(0, 1)
            eps = rng.normal(0, 0.25)
            outcome = 2.0 + firm_fe + trend_slope * (y - years[0]) + year_fe + 1.2 * did + 0.15 * x_size + eps
            rows.append(
                {
                    "stock_code": f,
                    "year": y,
                    "treat": treat,
                    "size": x_size,
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)


def test_ch11_did_positive_effect_and_placebo():
    df = _make_panel()
    cfg = DIDConfig()
    data = prepare_did_data(df, policy_year=2021, cfg=cfg)

    main = run_did_regression(data, y_col="outcome", cfg=cfg, controls=["size"])
    assert main["method"] == "DID"
    assert main["beta_did"] > 0.7
    assert main["pvalue_did"] < 0.05

    placebo = run_placebo_test(data, y_col="outcome", cfg=cfg, fake_policy_year=2018, controls=["size"])
    assert abs(placebo["beta_did"]) < main["beta_did"]


def test_ch11_dynamic_and_placebo_group_outputs():
    df = _make_panel(seed=12)
    cfg = DIDConfig()
    data = prepare_did_data(df, policy_year=2021, cfg=cfg)

    dyn = run_dynamic_did(data, y_col="outcome", cfg=cfg, policy_year=2021, leads=3, lags=2, controls=["size"])
    coef_table = dyn["coef_table"]
    assert not coef_table.empty
    assert {"term", "coef", "se", "p_value"}.issubset(coef_table.columns)

    placebo_group = run_placebo_group(data, y_col="outcome", cfg=cfg, controls=["size"], n_iter=30, seed=7)
    assert len(placebo_group) == 30
    assert {"iter", "beta_did", "pvalue_did"}.issubset(placebo_group.columns)

    summary = summarize_models([
        {"label": "main", **run_did_regression(data, "outcome", cfg, controls=["size"])},
        {"label": "placebo", **run_placebo_test(data, "outcome", cfg, fake_policy_year=2018, controls=["size"])},
    ])
    assert len(summary) == 2
    assert {"label", "beta_did", "se_did", "pvalue_did", "nobs", "r2"}.issubset(summary.columns)

    fig = plot_parallel_trends(data, y_col="outcome", cfg=cfg, policy_year=2021)
    assert fig is not None
    fig.clf()


def test_ch11_prepare_did_missing_column_raises():
    cfg = DIDConfig()
    bad = pd.DataFrame({"stock_code": ["000001"], "year": [2021]})
    try:
        prepare_did_data(bad, policy_year=2021, cfg=cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
