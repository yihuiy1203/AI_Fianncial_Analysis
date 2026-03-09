from __future__ import annotations

from typing import Sequence

import pandas as pd


DEFAULT_FEATURES = [
    "current_ratio",
    "quick_ratio",
    "debt_ratio",
    "equity_ratio",
    "roe",
    "roa",
    "gross_margin",
    "net_margin",
    "core_profit_ratio",
    "non_recurring_ratio",
    "fcf",
    "cash_earnings_ratio",
    "operating_cf_share",
    "investing_cf_share",
    "financing_cf_share",
]


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing required columns: {miss}")


def handle_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out.groupby("year", dropna=False)[c].transform(lambda s: s.fillna(s.median()))
        out[c] = out[c].fillna(out[c].median())
    return out


def winsorize(df: pd.DataFrame, cols: list[str], q_low: float = 0.01, q_high: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        lo, hi = x.quantile([q_low, q_high])
        out[c] = x.clip(lo, hi)
    return out


def build_feature_matrix(
    panel_df: pd.DataFrame,
    label_df: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    use_features = list(feature_cols) if feature_cols is not None else list(DEFAULT_FEATURES)
    _require_cols(panel_df, ["stock_code", "year", *use_features], "panel_df")
    _require_cols(label_df, ["stock_code", "year", "is_st"], "label_df")

    left = panel_df.copy()
    right = label_df.copy()
    left["stock_code"] = left["stock_code"].astype(str).str.strip().str.zfill(6)
    right["stock_code"] = right["stock_code"].astype(str).str.strip().str.zfill(6)
    left["year"] = pd.to_numeric(left["year"], errors="coerce").astype("Int64")
    right["year"] = pd.to_numeric(right["year"], errors="coerce").astype("Int64")

    df = left.merge(right[["stock_code", "year", "is_st"]], on=["stock_code", "year"], how="inner")
    df = handle_missing(df, use_features)
    df = winsorize(df, use_features)

    X = df[use_features].copy()
    y = pd.to_numeric(df["is_st"], errors="coerce").fillna(0).astype(int)
    meta = df[["stock_code", "year"]].copy()
    return X, y, meta
