from __future__ import annotations

import numpy as np
import pandas as pd

from ifa.causal.psm import check_balance, estimate_att, estimate_propensity_score, match_nearest_neighbor
from ifa.causal.rdd import estimate_rdd_effect
from ifa.causal.robustness import compare_methods


def build_demo_data(seed: int = 2026, n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    running = rng.uniform(-2.0, 2.0, n)
    size = rng.normal(0.0, 1.0, n)
    leverage = rng.normal(0.0, 1.0, n)

    treated_psm = (0.8 * size - 0.5 * leverage + rng.normal(0, 0.8, n) > 0).astype(int)
    y_rdd = 2.0 + 0.6 * running + 1.9 * (running >= 0).astype(float) + rng.normal(0, 0.5, n)
    y_psm = 1.5 + 1.6 * treated_psm + 0.5 * size - 0.3 * leverage + rng.normal(0, 0.7, n)
    return pd.DataFrame(
        {
            "running": running,
            "treated": treated_psm,
            "size": size,
            "leverage": leverage,
            "y_rdd": y_rdd,
            "y_psm": y_psm,
        }
    )


def main() -> None:
    df = build_demo_data()

    rdd_result = estimate_rdd_effect(df, running_var="running", outcome_var="y_rdd", cutoff=0.0)

    scored = estimate_propensity_score(df, treat_col="treated", covariate_cols=["size", "leverage"])
    pairs = match_nearest_neighbor(scored, treat_col="treated", ps_col="pscore", caliper=0.08)
    balance = check_balance(scored, pairs, ["size", "leverage"], treat_col="treated")
    psm_result = estimate_att(scored, pairs, outcome_col="y_psm")

    did_placeholder = {
        "effect": 1.4,
        "se": 0.35,
        "ci_low": 1.4 - 1.96 * 0.35,
        "ci_high": 1.4 + 1.96 * 0.35,
        "p_value": 0.01,
        "note": "placeholder DID from ch11",
    }

    summary = compare_methods({"DID": did_placeholder, "RDD": rdd_result, "PSM": psm_result})

    print("== RDD ==")
    print(rdd_result)
    print("\n== PSM Balance ==")
    print(balance.to_string(index=False))
    print("\n== Method Summary ==")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
