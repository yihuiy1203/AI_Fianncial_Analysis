from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ifa.config import settings, setup_logging


def check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def load_financial_tables(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bs = pd.read_csv(raw_dir / "balance_sheet.csv")
    is_ = pd.read_csv(raw_dir / "income_statement.csv")
    cf = pd.read_csv(raw_dir / "cash_flow.csv")
    return bs, is_, cf


def run_quickcheck(
    bs: pd.DataFrame,
    is_: pd.DataFrame,
    cf: pd.DataFrame,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    check_columns(
        bs,
        ["stock_code", "report_date", "total_assets", "total_liabilities", "total_equity"],
        "balance_sheet",
    )
    check_columns(
        is_,
        ["stock_code", "report_date", "net_profit"],
        "income_statement",
    )
    check_columns(
        cf,
        ["stock_code", "report_date", "operating_cash_flow"],
        "cash_flow",
    )

    bs_num = bs.copy()
    for col in ["total_assets", "total_liabilities", "total_equity"]:
        bs_num[col] = pd.to_numeric(bs_num[col], errors="coerce")

    is_num = is_.copy()
    is_num["net_profit"] = pd.to_numeric(is_num["net_profit"], errors="coerce")
    cf_num = cf.copy()
    cf_num["operating_cash_flow"] = pd.to_numeric(cf_num["operating_cash_flow"], errors="coerce")

    bs_missing = (
        bs_num[["total_assets", "total_liabilities", "total_equity"]]
        .isna()
        .sum()
        .to_dict()
    )

    valid_bs = bs_num.dropna(subset=["total_assets", "total_liabilities", "total_equity"])
    gap = (
        valid_bs["total_assets"]
        - valid_bs["total_liabilities"]
        - valid_bs["total_equity"]
    ).abs()
    accounting_pass_ratio = float((gap <= tolerance).mean()) if not gap.empty else float("nan")

    merged = is_num[["stock_code", "report_date", "net_profit"]].merge(
        cf_num[["stock_code", "report_date", "operating_cash_flow"]],
        on=["stock_code", "report_date"],
        how="inner",
    )
    merged = merged.dropna(subset=["net_profit", "operating_cash_flow"])
    if merged.empty:
        direction_consistency_ratio = float("nan")
    else:
        direction_consistency_ratio = float(
            ((merged["net_profit"] >= 0) == (merged["operating_cash_flow"] >= 0)).mean()
        )

    return {
        "bs_shape": tuple(bs.shape),
        "is_shape": tuple(is_.shape),
        "cf_shape": tuple(cf.shape),
        "bs_missing": bs_missing,
        "accounting_pass_ratio": accounting_pass_ratio,
        "direction_consistency_ratio": direction_consistency_ratio,
        "n_merged": int(merged.shape[0]),
    }


def run_quickcheck_from_dir(raw_dir: Path) -> dict[str, Any]:
    bs, is_, cf = load_financial_tables(raw_dir)
    return run_quickcheck(bs, is_, cf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 1 quick financial data checks")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=settings.get_path("data_raw"),
        help="Directory containing balance_sheet.csv, income_statement.csv, cash_flow.csv",
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("ch01.quickcheck")
    result = run_quickcheck_from_dir(args.raw_dir)

    logger.info("BS shape=%s, IS shape=%s, CF shape=%s", result["bs_shape"], result["is_shape"], result["cf_shape"])
    logger.info("BS missing=%s", result["bs_missing"])
    logger.info("Accounting pass ratio: %.2f%%", result["accounting_pass_ratio"] * 100)
    logger.info("Profit-CFO sign consistency: %.2f%%", result["direction_consistency_ratio"] * 100)
    logger.info("Merged sample size: %d", result["n_merged"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
