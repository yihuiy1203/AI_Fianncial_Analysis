from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _normal_two_sided_p_value(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def _as_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def select_bandwidth(running: pd.Series) -> float:
    x = pd.to_numeric(running, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size < 10:
        raise ValueError("At least 10 valid observations are required for bandwidth selection.")
    std = float(np.std(x, ddof=1))
    if std <= 0:
        raise ValueError("Running variable must have non-zero variance.")
    return 1.84 * std * (x.size ** (-1.0 / 5.0))


def fit_local_linear(
    x_centered: pd.Series,
    y: pd.Series,
    bandwidth: float,
) -> dict[str, Any]:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    x_arr = pd.to_numeric(x_centered, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]

    in_window = np.abs(x_arr) <= bandwidth
    xw = x_arr[in_window]
    yw = y_arr[in_window]
    if xw.size < 20:
        raise ValueError("Not enough observations inside bandwidth window.")

    d = (xw >= 0.0).astype(float)
    X = np.column_stack(
        [
            np.ones_like(xw),
            xw,
            d,
            xw * d,
        ]
    )
    weights = 1.0 - (np.abs(xw) / bandwidth)
    weights = np.clip(weights, 0.0, None)

    sqrt_w = np.sqrt(weights)
    Xw = X * sqrt_w[:, None]
    yw_w = yw * sqrt_w
    xtx = Xw.T @ Xw
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (Xw.T @ yw_w)
    resid = yw - X @ beta

    # HC1 robust covariance for weighted least squares.
    xu = X * (weights * resid)[:, None]
    meat = xu.T @ xu
    n = X.shape[0]
    k = X.shape[1]
    cov = (n / max(n - k, 1)) * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.diag(cov))

    tau = float(beta[2])
    se_tau = float(se[2]) if np.isfinite(se[2]) else float("nan")
    z = tau / se_tau if se_tau > 0 and np.isfinite(se_tau) else float("nan")
    p_value = _normal_two_sided_p_value(z) if np.isfinite(z) else float("nan")

    return {
        "effect": tau,
        "se": se_tau,
        "p_value": p_value,
        "ci_low": tau - 1.96 * se_tau if np.isfinite(se_tau) else float("nan"),
        "ci_high": tau + 1.96 * se_tau if np.isfinite(se_tau) else float("nan"),
        "n_window": int(xw.size),
        "weights": weights,
        "x_window": xw,
        "y_window": yw,
    }


def estimate_rdd_effect(
    df: pd.DataFrame,
    running_var: str,
    outcome_var: str,
    cutoff: float,
    bandwidth: float | None = None,
) -> dict[str, Any]:
    x_raw = _as_numeric_series(df, running_var)
    y = _as_numeric_series(df, outcome_var)
    data = pd.DataFrame({"x": x_raw, "y": y}).dropna()
    if data.empty:
        raise ValueError("No valid rows remain after numeric conversion.")

    x_centered = data["x"] - float(cutoff)
    bw = float(bandwidth) if bandwidth is not None else select_bandwidth(x_centered)
    fit = fit_local_linear(x_centered, data["y"], bw)
    return {
        "method": "RDD",
        "effect": fit["effect"],
        "se": fit["se"],
        "p_value": fit["p_value"],
        "ci_low": fit["ci_low"],
        "ci_high": fit["ci_high"],
        "bandwidth": bw,
        "n_window": fit["n_window"],
        "note": "local effect near cutoff",
    }


def plot_rdd(
    df: pd.DataFrame,
    running_var: str,
    outcome_var: str,
    cutoff: float,
    bandwidth: float | None = None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_rdd requires matplotlib.") from exc

    result = estimate_rdd_effect(
        df=df,
        running_var=running_var,
        outcome_var=outcome_var,
        cutoff=cutoff,
        bandwidth=bandwidth,
    )
    bw = result["bandwidth"]

    x = _as_numeric_series(df, running_var) - float(cutoff)
    y = _as_numeric_series(df, outcome_var)
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    win = data["x"].abs() <= bw
    sub = data.loc[win].copy()
    sub["side"] = np.where(sub["x"] >= 0.0, "right", "left")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(sub["x"], sub["y"], alpha=0.5, s=16)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("RDD Discontinuity Plot")
    ax.set_xlabel(f"{running_var} - cutoff")
    ax.set_ylabel(outcome_var)
    return fig, ax, result
