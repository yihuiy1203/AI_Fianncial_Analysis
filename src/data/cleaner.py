from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ifa.config import settings


def _read_required(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required raw file: {path}")
    return pd.read_csv(path)


def _winsorize(series: pd.Series, low: float = 0.01, high: float = 0.99) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    valid = x.dropna()
    if valid.empty:
        return x
    lo = float(valid.quantile(low))
    hi = float(valid.quantile(high))
    return x.clip(lower=lo, upper=hi)


def clean_tables(
    bs: pd.DataFrame,
    is_: pd.DataFrame,
    cf: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    bs_clean = bs.copy()
    is_clean = is_.copy()
    cf_clean = cf.copy()

    for df in [bs_clean, is_clean, cf_clean]:
        df.columns = [c.strip().lower() for c in df.columns]
        df["stock_code"] = df["stock_code"].astype(str).str.strip()
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df["year"] = df["report_date"].dt.year

    for col in [
        "total_assets",
        "total_liabilities",
        "total_equity",
        "current_assets",
        "non_current_assets",
        "current_liabilities",
        "inventory",
    ]:
        bs_clean[col] = _winsorize(bs_clean[col])
    for col in [
        "revenue",
        "cost",
        "operating_profit",
        "ebit",
        "profit_before_tax",
        "non_recurring_gain_loss",
        "net_profit",
    ]:
        is_clean[col] = _winsorize(is_clean[col])
    for col in [
        "operating_cash_flow",
        "investing_cash_flow",
        "financing_cash_flow",
        "capital_expenditure",
    ]:
        cf_clean[col] = _winsorize(cf_clean[col])

    bs_clean = bs_clean.dropna(
        subset=[
            "stock_code",
            "report_date",
            "year",
            "total_assets",
            "total_liabilities",
            "total_equity",
            "current_assets",
            "non_current_assets",
            "current_liabilities",
            "inventory",
        ]
    )
    is_clean = is_clean.dropna(
        subset=[
            "stock_code",
            "report_date",
            "year",
            "revenue",
            "cost",
            "operating_profit",
            "ebit",
            "profit_before_tax",
            "non_recurring_gain_loss",
            "net_profit",
        ]
    )
    cf_clean = cf_clean.dropna(
        subset=[
            "stock_code",
            "report_date",
            "year",
            "operating_cash_flow",
            "investing_cash_flow",
            "financing_cash_flow",
            "capital_expenditure",
        ]
    )

    bs_clean = bs_clean.sort_values(["stock_code", "report_date"]).drop_duplicates(["stock_code", "report_date"])
    is_clean = is_clean.sort_values(["stock_code", "report_date"]).drop_duplicates(["stock_code", "report_date"])
    cf_clean = cf_clean.sort_values(["stock_code", "report_date"]).drop_duplicates(["stock_code", "report_date"])

    return {
        "balance_sheet": bs_clean.reset_index(drop=True),
        "income_statement": is_clean.reset_index(drop=True),
        "cash_flow": cf_clean.reset_index(drop=True),
    }


def run_cleaner(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    bs = _read_required(input_dir, "balance_sheet.csv")
    is_ = _read_required(input_dir, "income_statement.csv")
    cf = _read_required(input_dir, "cash_flow.csv")

    before_missing = {
        "balance_sheet": bs.isna().sum().to_dict(),
        "income_statement": is_.isna().sum().to_dict(),
        "cash_flow": cf.isna().sum().to_dict(),
    }
    cleaned = clean_tables(bs, is_, cf)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in cleaned.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    after_missing = {name: df.isna().sum().to_dict() for name, df in cleaned.items()}
    return {
        "input_rows": {
            "balance_sheet": int(len(bs)),
            "income_statement": int(len(is_)),
            "cash_flow": int(len(cf)),
        },
        "output_rows": {name: int(len(df)) for name, df in cleaned.items()},
        "before_missing": before_missing,
        "after_missing": after_missing,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 2 data cleaner")
    parser.add_argument("--input", type=Path, default=settings.get_path("data_raw"))
    parser.add_argument("--output", type=Path, default=settings.get_path("data_cleaned"))
    args = parser.parse_args()

    summary = run_cleaner(args.input, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
