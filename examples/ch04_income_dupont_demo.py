from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from ifa.data.cleaner import run_cleaner
from ifa.data.crawler import run_crawler
from ifa.indicators.dupont import build_dupont_features
from ifa.indicators.income_statement import build_income_features


def main() -> None:
    base = Path(__file__).resolve().parent / f"_tmp_ch04_demo_{uuid.uuid4().hex}"
    raw_dir = base / "raw"
    cleaned_dir = base / "cleaned"
    base.mkdir(parents=True, exist_ok=False)
    try:
        crawl = run_crawler(
            codes=["000001", "000002", "000063"],
            years=[2019, 2020, 2021, 2022],
            output_dir=raw_dir,
        )
        run_cleaner(raw_dir, cleaned_dir)
        income_df = pd.read_csv(cleaned_dir / "income_statement.csv")
        bs_df = pd.read_csv(cleaned_dir / "balance_sheet.csv")

        merged = income_df.merge(
            bs_df[["stock_code", "year", "total_assets", "total_equity"]],
            on=["stock_code", "year"],
            how="inner",
        )
        income_feat = build_income_features(
            merged.assign(
                avg_total_assets=merged["total_assets"],
                avg_equity=merged["total_equity"],
            )
        )
        dup = build_dupont_features(income_df, bs_df)

        print(
            json.dumps(
                {
                    "crawl_rows": {
                        "income_statement": crawl["income_statement_rows"],
                        "balance_sheet": crawl["balance_sheet_rows"],
                    },
                    "income_feature_shape": list(income_feat.shape),
                    "dupont_shape": list(dup.shape),
                    "income_feature_cols": [
                        "gross_margin",
                        "net_margin",
                        "roe",
                        "roa",
                        "core_profit_ratio",
                        "non_recurring_ratio",
                        "revenue_yoy",
                        "profit_yoy",
                    ],
                    "dupont_cols": [
                        "tax_burden",
                        "interest_burden",
                        "operating_margin",
                        "asset_turnover",
                        "equity_multiplier",
                        "roe_dupont",
                        "roe_diff",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
