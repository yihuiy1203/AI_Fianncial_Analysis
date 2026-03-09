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


def calc_cash_flow_structure(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(
        df,
        ["stock_code", "year", "operating_cash_flow", "investing_cash_flow", "financing_cash_flow"],
        "cash_flow",
    )
    out = df.copy()
    denom = (
        pd.to_numeric(out["operating_cash_flow"], errors="coerce").abs()
        + pd.to_numeric(out["investing_cash_flow"], errors="coerce").abs()
        + pd.to_numeric(out["financing_cash_flow"], errors="coerce").abs()
    ).replace(0, pd.NA)
    out["operating_cf_share"] = _safe_div(out["operating_cash_flow"], denom)
    out["investing_cf_share"] = _safe_div(out["investing_cash_flow"], denom)
    out["financing_cf_share"] = _safe_div(out["financing_cash_flow"], denom)
    return out


def calc_fcf(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["stock_code", "year", "operating_cash_flow", "capital_expenditure"], "cash_flow")
    out = df.copy()
    out["fcf"] = pd.to_numeric(out["operating_cash_flow"], errors="coerce") - pd.to_numeric(
        out["capital_expenditure"], errors="coerce"
    )
    return out


def calc_cash_earnings_ratio(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["stock_code", "year", "operating_cash_flow", "net_profit"], "cash_flow")
    out = df.copy()
    out["cash_earnings_ratio"] = _safe_div(out["operating_cash_flow"], out["net_profit"])
    return out
