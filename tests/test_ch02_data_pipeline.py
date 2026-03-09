from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from ifa.data.cleaner import run_cleaner
from ifa.data.crawler import run_crawler
from ifa.data.loader import load, load_all


def _mk_tmp_dir() -> Path:
    d = Path(__file__).resolve().parent / f"_tmp_ch02_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def test_run_crawler_writes_expected_rows():
    base = _mk_tmp_dir()
    try:
        out = run_crawler(
            codes=["000001", "000002", "000063"],
            years=[2019, 2020, 2021],
            output_dir=base / "raw",
        )
        assert out["expected_rows_per_table"] == 9
        assert out["balance_sheet_rows"] == 9
        assert out["income_statement_rows"] == 9
        assert out["cash_flow_rows"] == 9
        assert out["success_rate"] == pytest.approx(1.0)
        assert (base / "raw" / "balance_sheet.csv").exists()
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_run_cleaner_and_loader_end_to_end():
    base = _mk_tmp_dir()
    raw_dir = base / "raw"
    cleaned_dir = base / "cleaned"
    try:
        run_crawler(
            codes=["000001", "000002"],
            years=[2020, 2021, 2022],
            output_dir=raw_dir,
        )
        clean_summary = run_cleaner(raw_dir, cleaned_dir)
        assert clean_summary["output_rows"]["balance_sheet"] == 6
        assert clean_summary["output_rows"]["income_statement"] == 6
        assert clean_summary["output_rows"]["cash_flow"] == 6

        df = load("000001", 2020, 2021, cleaned_dir=cleaned_dir)
        assert not df.empty
        assert df["stock_code"].nunique() == 1
        assert df["year"].between(2020, 2021).all()
        assert {"total_assets", "net_profit", "operating_cash_flow"}.issubset(df.columns)

        panel = load_all(cleaned_dir=cleaned_dir)
        assert len(panel) == 6
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_loader_raises_for_bad_year_range():
    with pytest.raises(ValueError, match="start_year must be <="):
        load("000001", 2023, 2022, cleaned_dir=Path("unused"))
