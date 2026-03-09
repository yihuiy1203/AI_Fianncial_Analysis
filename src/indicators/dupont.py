from __future__ import annotations

import pandas as pd


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    d = pd.to_numeric(denom, errors="coerce").replace(0, pd.NA)
    n = pd.to_numeric(numer, errors="coerce")
    return n / d


def _ensure_average_fields(bs_df: pd.DataFrame) -> pd.DataFrame:
    out = bs_df.copy()
    out = out.sort_values(["stock_code", "year"])
    if "avg_total_assets" not in out.columns:
        prev = out.groupby("stock_code", sort=False)["total_assets"].shift(1)
        out["avg_total_assets"] = (pd.to_numeric(out["total_assets"], errors="coerce") + prev) / 2.0
        out["avg_total_assets"] = out["avg_total_assets"].fillna(pd.to_numeric(out["total_assets"], errors="coerce"))
    if "avg_equity" not in out.columns:
        prev = out.groupby("stock_code", sort=False)["total_equity"].shift(1)
        out["avg_equity"] = (pd.to_numeric(out["total_equity"], errors="coerce") + prev) / 2.0
        out["avg_equity"] = out["avg_equity"].fillna(pd.to_numeric(out["total_equity"], errors="coerce"))
    return out


def merge_income_balance(income_df: pd.DataFrame, bs_df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(
        income_df,
        ["stock_code", "year", "net_profit", "profit_before_tax", "ebit", "revenue"],
        "income_df",
    )
    _require_cols(bs_df, ["stock_code", "year", "total_assets", "total_equity"], "bs_df")
    bs_use = _ensure_average_fields(bs_df)
    _require_cols(bs_use, ["avg_total_assets", "avg_equity"], "bs_df")
    income_use = income_df.drop(columns=["avg_total_assets", "avg_equity"], errors="ignore")
    merged = income_use.merge(
        bs_use[["stock_code", "year", "avg_total_assets", "avg_equity"]],
        on=["stock_code", "year"],
        how="inner",
    )
    return merged


def decompose_roe(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(
        df,
        ["net_profit", "profit_before_tax", "ebit", "revenue", "avg_total_assets", "avg_equity"],
        "dupont",
    )
    out = df.copy()
    out["tax_burden"] = _safe_div(out["net_profit"], out["profit_before_tax"])
    out["interest_burden"] = _safe_div(out["profit_before_tax"], out["ebit"])
    out["operating_margin"] = _safe_div(out["ebit"], out["revenue"])
    out["asset_turnover"] = _safe_div(out["revenue"], out["avg_total_assets"])
    out["equity_multiplier"] = _safe_div(out["avg_total_assets"], out["avg_equity"])
    out["roe_dupont"] = (
        out["tax_burden"]
        * out["interest_burden"]
        * out["operating_margin"]
        * out["asset_turnover"]
        * out["equity_multiplier"]
    )
    return out


def build_dupont_features(income_df: pd.DataFrame, bs_df: pd.DataFrame) -> pd.DataFrame:
    merged = merge_income_balance(income_df, bs_df)
    dup = decompose_roe(merged)
    dup["roe"] = _safe_div(dup["net_profit"], dup["avg_equity"])
    dup["roe_diff"] = (dup["roe_dupont"] - dup["roe"]).abs()
    return dup
