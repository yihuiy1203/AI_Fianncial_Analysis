from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventStudyConfig:
    est_start: int = -250
    est_end: int = -30
    event_start: int = -5
    event_end: int = 10
    min_est_obs: int = 120
    conf_level: float = 0.95


def _normal_two_sided_p_value(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _two_sided_binom_pvalue(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return float("nan")
    # Exact two-sided sign test p-value.
    prob_k = math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        prob_i = math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
        if prob_i <= prob_k + 1e-15:
            total += prob_i
    return float(min(1.0, total))


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing required columns: {miss}")


def build_event_panel(events: pd.DataFrame, returns: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    _require_columns(events, ["event_id", "stock_code", "event_date"], "events")
    _require_columns(returns, ["stock_code", "date", "daily_return"], "returns")
    _require_columns(market, ["date", "market_return"], "market")

    left = returns.copy()
    right = market.copy()
    evt = events.copy()

    left["stock_code"] = left["stock_code"].astype(str).str.strip().str.zfill(6)
    evt["stock_code"] = evt["stock_code"].astype(str).str.strip().str.zfill(6)
    left["date"] = pd.to_datetime(left["date"], errors="coerce")
    right["date"] = pd.to_datetime(right["date"], errors="coerce")
    evt["event_date"] = pd.to_datetime(evt["event_date"], errors="coerce")

    base = left.merge(right[["date", "market_return"]], on="date", how="inner")
    base = base.dropna(subset=["date"]).sort_values(["stock_code", "date"]).reset_index(drop=True)

    panels: list[pd.DataFrame] = []
    for _, row in evt.iterrows():
        code = row["stock_code"]
        event_id = row["event_id"]
        event_date = row["event_date"]
        if pd.isna(event_date):
            continue

        sec = base[base["stock_code"] == code].copy().reset_index(drop=True)
        if sec.empty:
            continue

        hit = sec.index[sec["date"] == event_date]
        if len(hit) == 0:
            # Align to first trading day on/after event date.
            future_hit = sec.index[sec["date"] > event_date]
            if len(future_hit) == 0:
                continue
            t0 = int(future_hit[0])
        else:
            t0 = int(hit[0])

        sec["t"] = sec.index - t0
        sec["event_id"] = event_id
        sec["event_date"] = event_date
        sec["event_day"] = sec.loc[t0, "date"]
        panels.append(sec)

    if not panels:
        return pd.DataFrame()
    out = pd.concat(panels, ignore_index=True)
    return out


def estimate_normal_return(event_df: pd.DataFrame, cfg: EventStudyConfig) -> tuple[float, float]:
    _require_columns(event_df, ["t", "daily_return", "market_return"], "event_df")
    est = event_df[(event_df["t"] >= cfg.est_start) & (event_df["t"] <= cfg.est_end)].copy()
    est["daily_return"] = pd.to_numeric(est["daily_return"], errors="coerce")
    est["market_return"] = pd.to_numeric(est["market_return"], errors="coerce")
    est = est.dropna(subset=["daily_return", "market_return"])

    if len(est) < cfg.min_est_obs:
        raise ValueError("Not enough observations in estimation window.")

    x = est["market_return"].to_numpy(dtype=float)
    y = est["daily_return"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def calc_abnormal_return(event_df: pd.DataFrame, alpha: float, beta: float, cfg: EventStudyConfig) -> pd.DataFrame:
    _require_columns(event_df, ["t", "daily_return", "market_return"], "event_df")
    win = event_df[(event_df["t"] >= cfg.event_start) & (event_df["t"] <= cfg.event_end)].copy()
    win = win.sort_values("t").reset_index(drop=True)
    win["daily_return"] = pd.to_numeric(win["daily_return"], errors="coerce")
    win["market_return"] = pd.to_numeric(win["market_return"], errors="coerce")
    win["expected_return"] = float(alpha) + float(beta) * win["market_return"]
    win["ar"] = win["daily_return"] - win["expected_return"]
    win["car"] = win["ar"].cumsum()

    cols = [
        "event_id",
        "stock_code",
        "date",
        "event_date",
        "event_day",
        "t",
        "daily_return",
        "market_return",
        "expected_return",
        "ar",
        "car",
    ]
    keep = [c for c in cols if c in win.columns]
    return win[keep]


def calc_car(ar_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(ar_df, ["event_id", "t", "ar"], "ar_df")
    out = ar_df.copy().sort_values(["event_id", "t"]).reset_index(drop=True)
    out["car"] = out.groupby("event_id")["ar"].cumsum()
    return out


def aggregate_acar(ar_panel: pd.DataFrame) -> pd.DataFrame:
    _require_columns(ar_panel, ["event_id", "t", "ar", "car"], "ar_panel")
    g = (
        ar_panel.groupby("t", as_index=False)
        .agg(aar=("ar", "mean"), acar=("car", "mean"), sd_car=("car", "std"), n=("event_id", "nunique"))
        .sort_values("t")
        .reset_index(drop=True)
    )
    g["se_car"] = g["sd_car"] / np.sqrt(g["n"].clip(lower=1))
    return g


def calc_acar(ar_panel: pd.DataFrame) -> pd.DataFrame:
    return aggregate_acar(ar_panel)


def test_significance(ar_panel: pd.DataFrame, cfg: EventStudyConfig) -> dict[str, float | int]:
    _require_columns(ar_panel, ["event_id", "t", "car"], "ar_panel")
    end_car = ar_panel[ar_panel["t"] == cfg.event_end][["event_id", "car"]].dropna()
    vals = end_car["car"].to_numpy(dtype=float)
    n = int(vals.size)
    if n == 0:
        return {
            "n": 0,
            "mean_car": float("nan"),
            "t_stat": float("nan"),
            "t_pvalue": float("nan"),
            "sign_pvalue": float("nan"),
        }

    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else float("nan")
    t_stat = mean / se if se and np.isfinite(se) else float("nan")
    t_p = _normal_two_sided_p_value(t_stat)

    pos = int((vals > 0).sum())
    sign_p = _two_sided_binom_pvalue(pos, n, p=0.5)
    return {
        "n": n,
        "mean_car": mean,
        "t_stat": float(t_stat),
        "t_pvalue": float(t_p),
        "sign_pvalue": float(sign_p),
    }


def run_event_study(
    events: pd.DataFrame,
    returns: pd.DataFrame,
    market: pd.DataFrame,
    cfg: EventStudyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    config = cfg or EventStudyConfig()
    panel = build_event_panel(events, returns, market)
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame(), {"n": 0}

    all_rows: list[pd.DataFrame] = []
    for _, one in panel.groupby("event_id"):
        try:
            alpha, beta = estimate_normal_return(one, config)
            one_ar = calc_abnormal_return(one, alpha, beta, config)
            one_ar["alpha"] = alpha
            one_ar["beta"] = beta
            all_rows.append(one_ar)
        except ValueError:
            continue

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame(), {"n": 0}

    ar_panel = pd.concat(all_rows, ignore_index=True)
    summary = aggregate_acar(ar_panel)
    stats = test_significance(ar_panel, config)
    return ar_panel, summary, stats


def plot_event_window(summary: pd.DataFrame, cfg: EventStudyConfig | None = None, title: str = "ACAR around event date"):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_event_window requires matplotlib.") from exc

    if summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No summary data", ha="center", va="center")
        ax.axis("off")
        return fig

    config = cfg or EventStudyConfig()
    z = NormalDist().inv_cdf(0.5 + config.conf_level / 2)
    ci = z * summary["se_car"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(summary["t"], summary["acar"], color="#1f77b4", linewidth=2, label="ACAR")
    ax.fill_between(summary["t"], summary["acar"] - ci, summary["acar"] + ci, color="#1f77b4", alpha=0.2)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Event Day (t=0)")
    ax.axhline(0, color="black", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Event Time (t)")
    ax.set_ylabel("ACAR")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
