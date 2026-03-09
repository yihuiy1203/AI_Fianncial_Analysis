from __future__ import annotations

import json

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
)


def build_demo_panel(seed: int = 2027) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firms = [f"{i:06d}" for i in range(1, 61)]
    years = list(range(2018, 2025))
    policy_year = 2021

    rows = []
    for i, stock in enumerate(firms):
        treat = 1 if i < 30 else 0
        firm_fe = rng.normal(0, 0.7)
        for y in years:
            post = 1 if y >= policy_year else 0
            did = treat * post
            size = rng.normal(0, 1)
            outcome = 1.0 + firm_fe + 0.08 * (y - years[0]) + 0.9 * did + 0.1 * size + rng.normal(0, 0.3)
            rows.append(
                {
                    "stock_code": stock,
                    "year": y,
                    "treat": treat,
                    "size": size,
                    "rd_intensity": outcome,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    policy_year = 2021
    cfg = DIDConfig()
    panel = build_demo_panel()
    data = prepare_did_data(panel, policy_year=policy_year, cfg=cfg)

    main_res = run_did_regression(data, y_col="rd_intensity", cfg=cfg, controls=["size"])
    dyn_res = run_dynamic_did(data, y_col="rd_intensity", cfg=cfg, policy_year=policy_year, leads=3, lags=3, controls=["size"])
    placebo_time = run_placebo_test(data, y_col="rd_intensity", cfg=cfg, fake_policy_year=2019, controls=["size"])
    placebo_group = run_placebo_group(data, y_col="rd_intensity", cfg=cfg, controls=["size"], n_iter=40, seed=8)

    fig = plot_parallel_trends(data, y_col="rd_intensity", cfg=cfg, policy_year=policy_year)
    fig.clf()

    print(
        json.dumps(
            {
                "n_rows": int(len(data)),
                "main": {
                    "beta_did": main_res["beta_did"],
                    "pvalue_did": main_res["pvalue_did"],
                    "nobs": main_res["nobs"],
                },
                "dynamic_terms": int(len(dyn_res["coef_table"])),
                "placebo_time_beta": placebo_time["beta_did"],
                "placebo_group_mean_beta": float(placebo_group["beta_did"].mean()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
