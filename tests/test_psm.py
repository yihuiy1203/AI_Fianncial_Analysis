import numpy as np
import pandas as pd

from ifa.causal.psm import (
    check_balance,
    estimate_att,
    estimate_propensity_score,
    match_nearest_neighbor,
)


def _make_psm_data(seed: int = 7, n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = rng.normal(0, 1, n)
    leverage = rng.normal(0, 1, n)
    latent = 0.9 * size - 0.7 * leverage + rng.normal(0, 0.8, n)
    treated = (latent > 0).astype(int)
    outcome = 3.0 + 1.8 * treated + 0.6 * size - 0.4 * leverage + rng.normal(0, 0.7, n)
    return pd.DataFrame(
        {
            "treated": treated,
            "size": size,
            "leverage": leverage,
            "outcome": outcome,
        }
    )


def test_psm_pipeline_recovers_positive_att():
    df = _make_psm_data()
    scored = estimate_propensity_score(df, "treated", ["size", "leverage"])
    pairs = match_nearest_neighbor(scored, caliper=0.08)
    assert not pairs.empty

    balance = check_balance(scored, pairs, ["size", "leverage"])
    assert (balance["smd_after"].abs() < balance["smd_before"].abs()).all()

    att = estimate_att(scored, pairs, "outcome")
    assert att["method"] == "PSM"
    assert att["n_pairs"] > 100
    assert 1.2 <= att["effect"] <= 2.4
    assert att["p_value"] < 0.05
