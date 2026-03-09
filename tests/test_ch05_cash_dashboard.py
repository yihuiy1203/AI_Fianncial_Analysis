from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from ifa.data.cleaner import run_cleaner
from ifa.data.crawler import run_crawler
from ifa.indicators.cash_flow import calc_cash_earnings_ratio, calc_cash_flow_structure, calc_fcf
from ifa.indicators.dashboard import build_full_panel, export_to_excel


def _sample_cash_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["000001", "000001", "000002"],
            "year": [2020, 2021, 2021],
            "operating_cash_flow": [30.0, 45.0, 20.0],
            "investing_cash_flow": [-20.0, -30.0, -10.0],
            "financing_cash_flow": [5.0, -8.0, 6.0],
            "capital_expenditure": [12.0, 16.0, 8.0],
            "net_profit": [18.0, 22.0, 14.0],
        }
    )


def test_cash_flow_indicator_functions():
    df = _sample_cash_df()
    s = calc_cash_flow_structure(df)
    f = calc_fcf(df)
    c = calc_cash_earnings_ratio(df)
    assert {"operating_cf_share", "investing_cf_share", "financing_cf_share"}.issubset(s.columns)
    assert f.loc[0, "fcf"] == pytest.approx(18.0)
    assert c.loc[0, "cash_earnings_ratio"] == pytest.approx(30.0 / 18.0)


def test_dashboard_build_and_export():
    base = Path(__file__).resolve().parent / f"_tmp_ch05_{uuid.uuid4().hex}"
    raw_dir = base / "raw"
    cleaned_dir = base / "cleaned"
    output_xlsx = base / "features" / "dashboard.xlsx"
    base.mkdir(parents=True, exist_ok=False)
    try:
        run_crawler(codes=["000001", "000002"], years=[2019, 2020, 2021], output_dir=raw_dir)
        run_cleaner(raw_dir, cleaned_dir)
        panel = build_full_panel("000001", 2019, 2021, cleaned_dir=cleaned_dir)
        assert not panel.empty
        for col in [
            "current_ratio",
            "net_margin",
            "roe_dupont",
            "operating_cf_share",
            "fcf",
            "cash_earnings_ratio",
        ]:
            assert col in panel.columns
        export_to_excel(panel, output_xlsx)
        assert output_xlsx.exists()
    finally:
        shutil.rmtree(base, ignore_errors=True)
