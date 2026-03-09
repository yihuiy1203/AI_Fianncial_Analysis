from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _normal_two_sided_p_value(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def estimate_propensity_score(
    df: pd.DataFrame,
    treat_col: str,
    covariate_cols: list[str],
    max_iter: int = 5000,
    lr: float = 0.05,
    l2: float = 1e-4,
) -> pd.DataFrame:
    work = df[[treat_col, *covariate_cols]].copy()
    for col in covariate_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work[treat_col] = pd.to_numeric(work[treat_col], errors="coerce")
    work = work.dropna()

    y = work[treat_col].astype(int).to_numpy(dtype=float)
    if not np.array_equal(np.unique(y), np.array([0.0, 1.0])):
        raise ValueError(f"{treat_col} must be binary 0/1.")

    X_raw = work[covariate_cols].to_numpy(dtype=float)
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    std[std == 0.0] = 1.0
    X_std = (X_raw - mean) / std
    X = np.column_stack([np.ones(X_std.shape[0]), X_std])

    beta = np.zeros(X.shape[1], dtype=float)
    n = X.shape[0]
    for _ in range(max_iter):
        p = _sigmoid(X @ beta)
        grad = (X.T @ (p - y)) / n
        reg = np.r_[0.0, beta[1:]] * l2
        beta -= lr * (grad + reg)

    pscore = _sigmoid(X @ beta)
    out = df.copy()
    out["pscore"] = np.nan
    out.loc[work.index, "pscore"] = pscore
    return out


def match_nearest_neighbor(
    df: pd.DataFrame,
    treat_col: str = "treated",
    ps_col: str = "pscore",
    caliper: float = 0.05,
) -> pd.DataFrame:
    if caliper <= 0:
        raise ValueError("caliper must be positive.")

    work = df[[treat_col, ps_col]].copy()
    work[treat_col] = pd.to_numeric(work[treat_col], errors="coerce")
    work[ps_col] = pd.to_numeric(work[ps_col], errors="coerce")
    work = work.dropna()

    treated = work[work[treat_col] == 1]
    control = work[work[treat_col] == 0]
    if treated.empty or control.empty:
        raise ValueError("Both treated and control observations are required.")

    c_scores = control[ps_col].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for t_idx, t_score in treated[ps_col].items():
        dist = np.abs(c_scores - t_score)
        best_pos = int(np.argmin(dist))
        best_dist = float(dist[best_pos])
        if best_dist <= caliper:
            rows.append(
                {
                    "t_idx": int(t_idx),
                    "c_idx": int(control.index[best_pos]),
                    "dist": best_dist,
                }
            )
    return pd.DataFrame(rows)


def _smd(x_t: pd.Series, x_c: pd.Series) -> float:
    m1 = float(x_t.mean())
    m0 = float(x_c.mean())
    v = (float(x_t.var(ddof=1)) + float(x_c.var(ddof=1))) / 2.0
    if not np.isfinite(v) or v <= 0:
        return 0.0
    return (m1 - m0) / math.sqrt(v)


def check_balance(
    df: pd.DataFrame,
    pair: pd.DataFrame,
    covariate_cols: list[str],
    treat_col: str = "treated",
) -> pd.DataFrame:
    if pair.empty:
        raise ValueError("pair is empty; no matched observations available.")

    treated_all = df[df[treat_col] == 1]
    control_all = df[df[treat_col] == 0]
    treated_matched = df.loc[pair["t_idx"]]
    control_matched = df.loc[pair["c_idx"]]

    rows = []
    for col in covariate_cols:
        t_before = pd.to_numeric(treated_all[col], errors="coerce").dropna()
        c_before = pd.to_numeric(control_all[col], errors="coerce").dropna()
        t_after = pd.to_numeric(treated_matched[col], errors="coerce").dropna()
        c_after = pd.to_numeric(control_matched[col], errors="coerce").dropna()
        rows.append(
            {
                "var": col,
                "smd_before": float(_smd(t_before, c_before)),
                "smd_after": float(_smd(t_after, c_after)),
            }
        )
    return pd.DataFrame(rows)


def estimate_att(
    df: pd.DataFrame,
    pair: pd.DataFrame,
    outcome_col: str,
) -> dict[str, Any]:
    if pair.empty:
        raise ValueError("pair is empty; cannot estimate ATT.")

    yt = pd.to_numeric(df.loc[pair["t_idx"], outcome_col], errors="coerce").to_numpy(dtype=float)
    yc = pd.to_numeric(df.loc[pair["c_idx"], outcome_col], errors="coerce").to_numpy(dtype=float)
    diff = yt - yc
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        raise ValueError("No valid matched outcome differences.")

    att = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else float("nan")
    z = att / se if se > 0 and np.isfinite(se) else float("nan")
    p_value = _normal_two_sided_p_value(z) if np.isfinite(z) else float("nan")
    return {
        "method": "PSM",
        "effect": att,
        "se": se,
        "p_value": p_value,
        "ci_low": att - 1.96 * se if np.isfinite(se) else float("nan"),
        "ci_high": att + 1.96 * se if np.isfinite(se) else float("nan"),
        "n_pairs": int(diff.size),
        "note": "ATT from nearest-neighbor matching",
    }
