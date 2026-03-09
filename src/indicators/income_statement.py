from __future__ import annotations

from typing import Any

import pandas as pd


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    d = pd.to_numeric(denom, errors="coerce").replace(0, pd.NA)
    n = pd.to_numeric(numer, errors="coerce")
    return n / d


def calc_profitability_ratios(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(
        df,
        ["stock_code", "year", "revenue", "cost", "net_profit", "avg_equity", "avg_total_assets"],
        "income_statement",
    )
    out = df.copy()
    out["gross_margin"] = _safe_div(out["revenue"] - out["cost"], out["revenue"])
    out["net_margin"] = _safe_div(out["net_profit"], out["revenue"])
    out["roe"] = _safe_div(out["net_profit"], out["avg_equity"])
    out["roa"] = _safe_div(out["net_profit"], out["avg_total_assets"])
    return out


def calc_earnings_quality(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(
        df,
        ["operating_profit", "net_profit", "non_recurring_gain_loss"],
        "income_statement",
    )
    out = df.copy()
    out["core_profit_ratio"] = _safe_div(out["operating_profit"], out["net_profit"])
    out["non_recurring_ratio"] = _safe_div(out["non_recurring_gain_loss"], out["net_profit"])
    return out


def calc_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["stock_code", "year", "revenue", "net_profit"], "income_statement")
    out = df.sort_values(["stock_code", "year"]).copy()
    out["revenue_yoy"] = out.groupby("stock_code", sort=False)["revenue"].pct_change()
    out["profit_yoy"] = out.groupby("stock_code", sort=False)["net_profit"].pct_change()
    return out


def build_income_features(income_df: pd.DataFrame) -> pd.DataFrame:
    out = calc_profitability_ratios(income_df)
    out = calc_earnings_quality(out)
    out = calc_growth_rates(out)
    return out


def quality_summary(panel: pd.DataFrame) -> dict[str, Any]:
    num = panel.select_dtypes("number")
    return {
        "missing_rate": panel.isna().mean().to_dict(),
        "extreme_count": (num.abs() > 100).sum().to_dict(),
    }
