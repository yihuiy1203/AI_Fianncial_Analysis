import numpy as np
import pandas as pd

from ifa.causal.rdd import estimate_rdd_effect, select_bandwidth


def test_select_bandwidth_positive():
    x = pd.Series(np.linspace(-3, 3, 200))
    bw = select_bandwidth(x)
    assert bw > 0


def test_estimate_rdd_effect_detects_jump():
    rng = np.random.default_rng(42)
    n = 1600
    x = rng.uniform(-2, 2, n)
    treatment = (x >= 0).astype(float)
    y = 1.5 + 0.8 * x + 2.2 * treatment + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"running": x, "outcome": y})

    result = estimate_rdd_effect(
        df=df,
        running_var="running",
        outcome_var="outcome",
        cutoff=0.0,
    )
    assert result["method"] == "RDD"
    assert result["n_window"] > 50
    assert 1.7 <= result["effect"] <= 2.7
    assert result["p_value"] < 0.05
