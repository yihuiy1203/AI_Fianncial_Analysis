from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from ifa.data.cleaner import run_cleaner
from ifa.data.crawler import run_crawler
from ifa.indicators.balance_sheet import (
    build_balance_sheet_panel,
    calc_asset_structure,
    calc_leverage_ratios,
    calc_liquidity_ratios,
    quality_summary,
    run_balance_sheet_pipeline,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["000001", "000001", "000002"],
            "year": [2020, 2021, 2021],
            "current_assets": [60.0, 80.0, 90.0],
            "inventory": [15.0, 20.0, 10.0],
            "current_liabilities": [30.0, 40.0, 45.0],
            "total_assets": [120.0, 150.0, 180.0],
            "total_liabilities": [70.0, 90.0, 100.0],
            "total_equity": [50.0, 60.0, 80.0],
            "non_current_assets": [60.0, 70.0, 90.0],
        }
    )


def test_calc_liquidity_ratios():
    out = calc_liquidity_ratios(_sample_df())
    assert "current_ratio" in out.columns
    assert "quick_ratio" in out.columns
    assert out.loc[0, "current_ratio"] == pytest.approx(2.0)
    assert out.loc[0, "quick_ratio"] == pytest.approx(1.5)


def test_calc_leverage_and_structure():
    lev = calc_leverage_ratios(_sample_df())
    ast = calc_asset_structure(_sample_df())
    assert lev.loc[0, "debt_ratio"] == pytest.approx(70 / 120)
    assert lev.loc[0, "equity_ratio"] == pytest.approx(50 / 120)
    assert ast.loc[0, "current_assets_share"] == pytest.approx(0.5)
    assert ast.loc[0, "non_current_assets_share"] == pytest.approx(0.5)


def test_build_panel_and_quality_summary():
    panel = build_balance_sheet_panel(_sample_df())
    for col in [
        "current_ratio",
        "quick_ratio",
        "debt_ratio",
        "equity_ratio",
        "current_assets_share",
        "non_current_assets_share",
    ]:
        assert col in panel.columns
    q = quality_summary(panel)
    assert "missing_rate" in q
    assert "extreme_count" in q
    assert q["structure_share_max_deviation"] == pytest.approx(0.0)


def test_run_balance_sheet_pipeline_end_to_end():
    base = Path(__file__).resolve().parent / f"_tmp_ch03_{uuid.uuid4().hex}"
    raw_dir = base / "raw"
    cleaned_dir = base / "cleaned"
    out_path = base / "features" / "balance_sheet_panel.csv"
    base.mkdir(parents=True, exist_ok=False)
    try:
        run_crawler(
            codes=["000001", "000002"],
            years=[2019, 2020, 2021],
            output_dir=raw_dir,
        )
        run_cleaner(raw_dir, cleaned_dir)
        panel = run_balance_sheet_pipeline(
            code="000001",
            start_year=2019,
            end_year=2021,
            output_path=out_path,
            cleaned_dir=cleaned_dir,
        )
        assert not panel.empty
        assert out_path.exists()
        assert panel["year"].between(2019, 2021).all()
    finally:
        shutil.rmtree(base, ignore_errors=True)
