from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from ifa.data.cleaner import run_cleaner
from ifa.data.crawler import run_crawler
from ifa.indicators.dupont import build_dupont_features
from ifa.indicators.income_statement import (
    build_income_features,
    calc_earnings_quality,
    calc_growth_rates,
    calc_profitability_ratios,
)


def _sample_income() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["000001", "000001", "000001", "000002"],
            "year": [2020, 2021, 2022, 2021],
            "revenue": [100.0, 120.0, 132.0, 90.0],
            "cost": [70.0, 82.0, 90.0, 60.0],
            "net_profit": [10.0, 12.0, 15.0, 9.0],
            "operating_profit": [14.0, 17.0, 20.0, 12.0],
            "ebit": [16.0, 19.0, 22.0, 13.0],
            "profit_before_tax": [14.0, 16.0, 19.0, 11.0],
            "non_recurring_gain_loss": [1.0, 0.8, 1.2, 0.5],
            "avg_equity": [80.0, 85.0, 90.0, 70.0],
            "avg_total_assets": [160.0, 170.0, 180.0, 140.0],
        }
    )


def _sample_bs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["000001", "000001", "000001", "000002"],
            "year": [2020, 2021, 2022, 2021],
            "total_assets": [160.0, 170.0, 180.0, 140.0],
            "total_equity": [80.0, 85.0, 90.0, 70.0],
        }
    )


def test_profitability_and_quality_metrics():
    income = _sample_income()
    p = calc_profitability_ratios(income)
    q = calc_earnings_quality(p)
    assert p.loc[0, "gross_margin"] == pytest.approx(0.3)
    assert p.loc[0, "net_margin"] == pytest.approx(0.1)
    assert p.loc[0, "roe"] == pytest.approx(10 / 80)
    assert p.loc[0, "roa"] == pytest.approx(10 / 160)
    assert q.loc[0, "core_profit_ratio"] == pytest.approx(14 / 10)
    assert q.loc[0, "non_recurring_ratio"] == pytest.approx(1 / 10)


def test_growth_rates_sorted_by_stock_and_year():
    income = _sample_income().sample(frac=1.0, random_state=42)
    g = calc_growth_rates(income)
    sub = g[g["stock_code"] == "000001"].sort_values("year").reset_index(drop=True)
    assert pd.isna(sub.loc[0, "revenue_yoy"])
    assert sub.loc[1, "revenue_yoy"] == pytest.approx(0.2)
    assert sub.loc[2, "profit_yoy"] == pytest.approx(0.25)


def test_dupont_matches_accounting_roe():
    income = _sample_income()
    bs = _sample_bs()
    dup = build_dupont_features(income, bs)
    assert {"roe_dupont", "roe", "roe_diff"}.issubset(dup.columns)
    assert (dup["roe_diff"] < 1e-12).all()


def test_end_to_end_ch04_pipeline():
    base = Path(__file__).resolve().parent / f"_tmp_ch04_{uuid.uuid4().hex}"
    raw_dir = base / "raw"
    cleaned_dir = base / "cleaned"
    base.mkdir(parents=True, exist_ok=False)
    try:
        run_crawler(codes=["000001", "000002"], years=[2019, 2020, 2021], output_dir=raw_dir)
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
        assert not income_feat.empty
        assert not dup.empty
        assert {"gross_margin", "net_margin", "core_profit_ratio", "revenue_yoy", "profit_yoy"}.issubset(
            income_feat.columns
        )
        assert {"tax_burden", "interest_burden", "operating_margin", "asset_turnover", "equity_multiplier"}.issubset(
            dup.columns
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)
