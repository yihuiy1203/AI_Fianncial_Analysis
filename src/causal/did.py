from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DIDConfig:
    id_col: str = "stock_code"
    time_col: str = "year"
    treat_col: str = "treat"
    post_col: str = "post"
    did_col: str = "did"
    cluster_col: str = "stock_code"
    conf_level: float = 0.95


def _normal_two_sided_p_value(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing required columns: {miss}")


def prepare_did_data(
    df: pd.DataFrame,
    policy_year: int,
    cfg: DIDConfig,
    treat_col: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    src_treat = treat_col or cfg.treat_col
    _require_cols(out, [cfg.id_col, cfg.time_col, src_treat], "df")

    out[cfg.id_col] = out[cfg.id_col].astype(str).str.strip().str.zfill(6)
    out[cfg.time_col] = pd.to_numeric(out[cfg.time_col], errors="coerce").astype("Int64")
    out[cfg.treat_col] = pd.to_numeric(out[src_treat], errors="coerce").fillna(0).astype(int)

    out[cfg.post_col] = (out[cfg.time_col] >= int(policy_year)).astype(int)
    out[cfg.did_col] = out[cfg.treat_col] * out[cfg.post_col]
    return out


def _build_design_matrix(
    df: pd.DataFrame,
    cfg: DIDConfig,
    y_col: str,
    controls: list[str] | None,
    entity_fe: bool,
    time_fe: bool,
    extra_terms: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    use_controls = controls or []
    use_extra = extra_terms or []
    _require_cols(df, [cfg.id_col, cfg.time_col, y_col, cfg.treat_col, cfg.post_col, cfg.did_col, *use_controls, *use_extra], "df")

    work = df[[cfg.id_col, cfg.time_col, y_col, cfg.treat_col, cfg.post_col, cfg.did_col, *use_controls, *use_extra]].copy()
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    for c in [cfg.treat_col, cfg.post_col, cfg.did_col, *use_controls, *use_extra]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[y_col, cfg.treat_col, cfg.post_col, cfg.did_col, *use_controls, *use_extra])
    if work.empty:
        raise ValueError("No valid rows after numeric conversion.")

    cols: list[np.ndarray] = [np.ones(len(work), dtype=float)]
    names: list[str] = ["Intercept", cfg.treat_col, cfg.post_col, cfg.did_col]
    cols.extend(
        [
            work[cfg.treat_col].to_numpy(dtype=float),
            work[cfg.post_col].to_numpy(dtype=float),
            work[cfg.did_col].to_numpy(dtype=float),
        ]
    )

    for c in use_controls:
        cols.append(work[c].to_numpy(dtype=float))
        names.append(c)

    for c in use_extra:
        cols.append(work[c].to_numpy(dtype=float))
        names.append(c)

    if entity_fe:
        d_id = pd.get_dummies(work[cfg.id_col], prefix="id", drop_first=True)
        for c in d_id.columns:
            cols.append(d_id[c].to_numpy(dtype=float))
            names.append(str(c))

    if time_fe:
        d_t = pd.get_dummies(work[cfg.time_col].astype(str), prefix="yr", drop_first=True)
        for c in d_t.columns:
            cols.append(d_t[c].to_numpy(dtype=float))
            names.append(str(c))

    X = np.column_stack(cols)
    y = work[y_col].to_numpy(dtype=float)
    return X, y, names, work


def _ols_cluster(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    if clusters is None:
        xu = X * resid[:, None]
        meat = xu.T @ xu
        cov = (n / max(n - k, 1)) * (xtx_inv @ meat @ xtx_inv)
        return beta, cov

    cl = pd.Series(clusters).astype(str).to_numpy()
    uniq = pd.unique(cl)
    g = len(uniq)
    meat = np.zeros((k, k), dtype=float)
    for u in uniq:
        idx = np.where(cl == u)[0]
        Xg = X[idx, :]
        ug = resid[idx]
        xugu = Xg.T @ ug
        meat += np.outer(xugu, xugu)

    scale = 1.0
    if g > 1:
        scale *= g / (g - 1)
    if n > k:
        scale *= (n - 1) / (n - k)

    cov = scale * (xtx_inv @ meat @ xtx_inv)
    return beta, cov


def run_did_regression(
    df: pd.DataFrame,
    y_col: str,
    cfg: DIDConfig,
    controls: list[str] | None = None,
    entity_fe: bool = True,
    time_fe: bool = True,
) -> dict[str, Any]:
    X, y, names, work = _build_design_matrix(df, cfg, y_col, controls, entity_fe, time_fe)

    clusters = work[cfg.cluster_col].to_numpy() if cfg.cluster_col in work.columns else None
    beta, cov = _ols_cluster(X, y, clusters)
    se = np.sqrt(np.diag(cov))

    idx = names.index(cfg.did_col)
    b = float(beta[idx])
    s = float(se[idx]) if np.isfinite(se[idx]) else float("nan")
    z = b / s if s > 0 and np.isfinite(s) else float("nan")
    p = _normal_two_sided_p_value(z)

    y_hat = X @ beta
    sst = float(np.sum((y - np.mean(y)) ** 2))
    ssr = float(np.sum((y - y_hat) ** 2))
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")

    return {
        "method": "DID",
        "effect": b,
        "se": s,
        "p_value": p,
        "ci_low": b - 1.96 * s if np.isfinite(s) else float("nan"),
        "ci_high": b + 1.96 * s if np.isfinite(s) else float("nan"),
        "beta_did": b,
        "se_did": s,
        "pvalue_did": p,
        "nobs": int(len(y)),
        "r2": float(r2),
        "formula": f"{y_col} ~ treat + post + did + FE({cfg.id_col},{cfg.time_col})",
    }


def plot_parallel_trends(df: pd.DataFrame, y_col: str, cfg: DIDConfig, policy_year: int):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_parallel_trends requires matplotlib.") from exc

    _require_cols(df, [cfg.time_col, cfg.treat_col, y_col], "df")
    grp = (
        df.groupby([cfg.time_col, cfg.treat_col], as_index=False)[y_col]
        .mean()
        .rename(columns={y_col: "mean_y"})
        .sort_values(cfg.time_col)
    )
    treat = grp[grp[cfg.treat_col] == 1]
    ctrl = grp[grp[cfg.treat_col] == 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ctrl[cfg.time_col], ctrl["mean_y"], marker="o", label="Control")
    ax.plot(treat[cfg.time_col], treat["mean_y"], marker="o", label="Treated")
    ax.axvline(policy_year, color="red", linestyle="--", linewidth=1.2, label="Policy Year")
    ax.set_title(f"Parallel Trends Check: {y_col}")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Mean {y_col}")
    ax.legend()
    fig.tight_layout()
    return fig


def run_dynamic_did(
    df: pd.DataFrame,
    y_col: str,
    cfg: DIDConfig,
    policy_year: int,
    leads: int = 3,
    lags: int = 3,
    controls: list[str] | None = None,
) -> dict[str, Any]:
    data = df.copy()
    data["rel_year"] = pd.to_numeric(data[cfg.time_col], errors="coerce") - int(policy_year)
    data["rel_year"] = data["rel_year"].clip(lower=-leads, upper=lags)

    event_terms: list[str] = []
    for k in range(-leads, lags + 1):
        if k == -1:
            continue
        name = f"evt_{k}"
        data[name] = ((data["rel_year"] == k).astype(int) * data[cfg.treat_col]).astype(int)
        event_terms.append(name)

    X, y, names, work = _build_design_matrix(
        data,
        cfg,
        y_col,
        controls,
        entity_fe=True,
        time_fe=True,
        extra_terms=event_terms,
    )
    beta, cov = _ols_cluster(X, y, work[cfg.cluster_col].to_numpy() if cfg.cluster_col in work.columns else None)
    se = np.sqrt(np.diag(cov))

    rows = []
    for term in event_terms:
        i = names.index(term)
        b = float(beta[i])
        s = float(se[i]) if np.isfinite(se[i]) else float("nan")
        z = b / s if s > 0 and np.isfinite(s) else float("nan")
        rows.append({"term": term, "coef": b, "se": s, "p_value": _normal_two_sided_p_value(z)})
    coef_df = pd.DataFrame(rows)

    pre_terms = coef_df[coef_df["term"].str.contains("evt_-")]
    pretrend_max_abs = float(pre_terms["coef"].abs().max()) if not pre_terms.empty else float("nan")

    return {
        "event_terms": event_terms,
        "coef_table": coef_df,
        "pretrend_max_abs": pretrend_max_abs,
        "nobs": int(len(y)),
    }


def run_placebo_test(
    df: pd.DataFrame,
    y_col: str,
    cfg: DIDConfig,
    fake_policy_year: int,
    controls: list[str] | None = None,
) -> dict[str, Any]:
    fake = df.copy()
    fake[cfg.post_col] = (pd.to_numeric(fake[cfg.time_col], errors="coerce") >= int(fake_policy_year)).astype(int)
    fake[cfg.did_col] = fake[cfg.treat_col] * fake[cfg.post_col]
    out = run_did_regression(fake, y_col, cfg, controls=controls)
    out["fake_year"] = int(fake_policy_year)
    return out


def run_placebo_group(
    df: pd.DataFrame,
    y_col: str,
    cfg: DIDConfig,
    controls: list[str] | None = None,
    n_iter: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = df[cfg.id_col].astype(str).drop_duplicates().to_numpy()
    n_treat = int(df.drop_duplicates(cfg.id_col)[cfg.treat_col].sum())

    rows: list[dict[str, Any]] = []
    for i in range(n_iter):
        fake_ids = set(rng.choice(ids, size=n_treat, replace=False))
        fake = df.copy()
        fake[cfg.treat_col] = fake[cfg.id_col].astype(str).isin(fake_ids).astype(int)
        fake[cfg.did_col] = fake[cfg.treat_col] * fake[cfg.post_col]
        out = run_did_regression(fake, y_col, cfg, controls=controls)
        rows.append({"iter": i, "beta_did": out["beta_did"], "pvalue_did": out["pvalue_did"]})
    return pd.DataFrame(rows)


def summarize_models(models: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for m in models:
        rows.append(
            {
                "label": m.get("label", m.get("method", "model")),
                "beta_did": float(m.get("beta_did", np.nan)),
                "se_did": float(m.get("se_did", np.nan)),
                "pvalue_did": float(m.get("pvalue_did", np.nan)),
                "nobs": int(m.get("nobs", 0)),
                "r2": float(m.get("r2", np.nan)),
            }
        )
    return pd.DataFrame(rows)
