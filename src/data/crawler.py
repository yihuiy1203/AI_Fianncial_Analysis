from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ifa.config import settings


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _rng_for(code: str, year: int) -> np.random.Generator:
    seed = (sum(ord(c) for c in code) * 131 + year * 17) % (2**32 - 1)
    return np.random.default_rng(seed)


def crawl_financial_tables(codes: list[str], years: list[int]) -> dict[str, pd.DataFrame]:
    bs_rows: list[dict[str, Any]] = []
    is_rows: list[dict[str, Any]] = []
    cf_rows: list[dict[str, Any]] = []

    for code in codes:
        for year in years:
            rng = _rng_for(code, year)
            assets = float(rng.uniform(80, 600))
            liabilities = float(assets * rng.uniform(0.25, 0.75))
            equity = assets - liabilities
            current_assets = float(assets * rng.uniform(0.3, 0.75))
            non_current_assets = assets - current_assets
            current_liabilities = float(liabilities * rng.uniform(0.35, 0.85))
            inventory = float(current_assets * rng.uniform(0.08, 0.4))
            net_profit = float(rng.normal(loc=assets * 0.04, scale=max(assets * 0.01, 1.0)))
            cfo = float(net_profit + rng.normal(loc=0.0, scale=max(assets * 0.008, 0.8)))
            capex = float(max(assets * rng.uniform(0.03, 0.14), 0.0))
            cfi = float(-capex + rng.normal(loc=0.0, scale=max(assets * 0.004, 0.3)))
            cff = float(rng.normal(loc=assets * 0.01, scale=max(assets * 0.007, 0.4)))
            revenue = float(assets * rng.uniform(0.35, 1.2))
            gross_margin = float(rng.uniform(0.12, 0.48))
            cost = revenue * (1.0 - gross_margin)
            operating_profit = revenue - cost - float(rng.uniform(0.02, 0.15) * revenue)
            ebit = operating_profit + float(rng.uniform(-0.01, 0.03) * revenue)
            profit_before_tax = ebit - float(rng.uniform(0.0, 0.04) * assets)
            non_recurring = float(rng.normal(loc=0.0, scale=max(abs(net_profit) * 0.08, 0.5)))
            # Keep accounting linkage stable for teaching data.
            net_profit = profit_before_tax * float(rng.uniform(0.72, 0.85))

            report_date = f"{year}-12-31"
            bs_rows.append(
                {
                    "stock_code": code,
                    "report_date": report_date,
                    "total_assets": round(assets, 4),
                    "total_liabilities": round(liabilities, 4),
                    "total_equity": round(equity, 4),
                    "current_assets": round(current_assets, 4),
                    "non_current_assets": round(non_current_assets, 4),
                    "current_liabilities": round(current_liabilities, 4),
                    "inventory": round(inventory, 4),
                }
            )
            is_rows.append(
                {
                    "stock_code": code,
                    "report_date": report_date,
                    "revenue": round(revenue, 4),
                    "cost": round(cost, 4),
                    "operating_profit": round(operating_profit, 4),
                    "ebit": round(ebit, 4),
                    "profit_before_tax": round(profit_before_tax, 4),
                    "non_recurring_gain_loss": round(non_recurring, 4),
                    "net_profit": round(net_profit, 4),
                }
            )
            cf_rows.append(
                {
                    "stock_code": code,
                    "report_date": report_date,
                    "operating_cash_flow": round(cfo, 4),
                    "investing_cash_flow": round(cfi, 4),
                    "financing_cash_flow": round(cff, 4),
                    "capital_expenditure": round(capex, 4),
                }
            )

    return {
        "balance_sheet": pd.DataFrame(bs_rows),
        "income_statement": pd.DataFrame(is_rows),
        "cash_flow": pd.DataFrame(cf_rows),
    }


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, df in tables.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = str(path)
    return paths


def run_crawler(codes: list[str], years: list[int], output_dir: Path) -> dict[str, Any]:
    if not codes:
        raise ValueError("codes must not be empty.")
    if not years:
        raise ValueError("years must not be empty.")

    tables = crawl_financial_tables(codes, years)
    paths = save_tables(tables, output_dir)
    expected = len(codes) * len(years)
    return {
        "n_codes": len(codes),
        "n_years": len(years),
        "expected_rows_per_table": expected,
        "balance_sheet_rows": int(len(tables["balance_sheet"])),
        "income_statement_rows": int(len(tables["income_statement"])),
        "cash_flow_rows": int(len(tables["cash_flow"])),
        "success_rate": 1.0,
        "output_paths": paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 2 financial data crawler (teaching edition)")
    parser.add_argument("--codes", type=str, default="000001,000002,000063")
    parser.add_argument("--years", type=str, default="2019,2020,2021")
    parser.add_argument("--output", type=Path, default=settings.get_path("data_raw"))
    args = parser.parse_args()

    codes = _parse_csv_arg(args.codes)
    years = [int(y) for y in _parse_csv_arg(args.years)]
    result = run_crawler(codes, years, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
