from __future__ import annotations

from itertools import product
from typing import Any, Callable

import pandas as pd


def compare_methods(results_dict: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for method, result in results_dict.items():
        rows.append(
            {
                "method": method,
                "effect": float(result["effect"]),
                "se": float(result["se"]),
                "ci_low": float(result["ci_low"]),
                "ci_high": float(result["ci_high"]),
                "p_value": float(result["p_value"]),
                "note": str(result.get("note", "")),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["sig_5pct"] = out["p_value"] < 0.05
    return out.sort_values("method").reset_index(drop=True)


def run_sensitivity_analysis(
    estimator: Callable[..., dict[str, Any]],
    param_grid: dict[str, list[Any]],
) -> pd.DataFrame:
    if not param_grid:
        raise ValueError("param_grid must not be empty.")

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    rows = []
    for combo in product(*values):
        params = dict(zip(keys, combo, strict=False))
        result = estimator(**params)
        rows.append(
            {
                **params,
                "effect": float(result["effect"]),
                "se": float(result["se"]),
                "ci_low": float(result["ci_low"]),
                "ci_high": float(result["ci_high"]),
                "p_value": float(result["p_value"]),
            }
        )
    return pd.DataFrame(rows)
