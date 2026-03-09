from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ifa.indicators.dupont import build_dupont_features
from ifa.indicators.income_statement import build_income_features


def main() -> None:
    income_df = pd.read_csv(Path("data/cleaned/income_statement.csv"))
    bs_df = pd.read_csv(Path("data/cleaned/balance_sheet.csv"))
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

    Path("data/features").mkdir(parents=True, exist_ok=True)
    income_feat.to_csv("data/features/profitability_features.csv", index=False)
    dup.to_csv("data/features/dupont_features.csv", index=False)
    print(
        json.dumps(
            {
                "income_features_rows": int(len(income_feat)),
                "dupont_rows": int(len(dup)),
                "outputs": ["data/features/profitability_features.csv", "data/features/dupont_features.csv"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
