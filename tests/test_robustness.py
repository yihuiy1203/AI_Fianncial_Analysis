from ifa.causal.robustness import compare_methods, run_sensitivity_analysis


def test_compare_methods_builds_summary():
    summary = compare_methods(
        {
            "RDD": {
                "effect": 1.1,
                "se": 0.3,
                "ci_low": 0.5,
                "ci_high": 1.7,
                "p_value": 0.01,
                "note": "local effect",
            },
            "PSM": {
                "effect": 0.9,
                "se": 0.4,
                "ci_low": 0.1,
                "ci_high": 1.7,
                "p_value": 0.03,
            },
        }
    )
    assert list(summary.columns) == [
        "method",
        "effect",
        "se",
        "ci_low",
        "ci_high",
        "p_value",
        "note",
        "sig_5pct",
    ]
    assert summary.shape[0] == 2
    assert summary["sig_5pct"].all()


def test_run_sensitivity_analysis_expands_grid():
    def estimator(alpha: float, beta: float):
        effect = alpha + beta
        se = 0.2
        return {
            "effect": effect,
            "se": se,
            "ci_low": effect - 1.96 * se,
            "ci_high": effect + 1.96 * se,
            "p_value": 0.04,
        }

    out = run_sensitivity_analysis(
        estimator=estimator,
        param_grid={"alpha": [0.1, 0.2], "beta": [1.0, 2.0]},
    )
    assert out.shape[0] == 4
    assert {"alpha", "beta", "effect", "se", "ci_low", "ci_high", "p_value"} == set(out.columns)
