from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest

from ifa.data.ch01_quickcheck import (
    check_columns,
    run_quickcheck,
    run_quickcheck_from_dir,
)


def _build_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bs = pd.DataFrame(
        [
            {
                "stock_code": "A",
                "report_date": "2024-12-31",
                "total_assets": 100,
                "total_liabilities": 60,
                "total_equity": 40,
            },
            {
                "stock_code": "B",
                "report_date": "2024-12-31",
                "total_assets": 200,
                "total_liabilities": 120,
                "total_equity": 75,  # one failed accounting identity
            },
            {
                "stock_code": "C",
                "report_date": "2024-12-31",
                "total_assets": 90,
                "total_liabilities": 40,
                "total_equity": 50,
            },
        ]
    )
    is_ = pd.DataFrame(
        [
            {"stock_code": "A", "report_date": "2024-12-31", "net_profit": 12},
            {"stock_code": "B", "report_date": "2024-12-31", "net_profit": -4},
            {"stock_code": "C", "report_date": "2024-12-31", "net_profit": 5},
        ]
    )
    cf = pd.DataFrame(
        [
            {"stock_code": "A", "report_date": "2024-12-31", "operating_cash_flow": 2},
            {"stock_code": "B", "report_date": "2024-12-31", "operating_cash_flow": -3},
            {"stock_code": "C", "report_date": "2024-12-31", "operating_cash_flow": -1},  # inconsistent sign
        ]
    )
    return bs, is_, cf


def test_check_columns_raises_for_missing_column():
    df = pd.DataFrame({"stock_code": ["A"], "report_date": ["2024-12-31"]})
    with pytest.raises(ValueError, match="missing required columns"):
        check_columns(df, ["stock_code", "report_date", "net_profit"], "income_statement")


def test_run_quickcheck_returns_expected_metrics():
    bs, is_, cf = _build_frames()
    out = run_quickcheck(bs, is_, cf)
    assert out["bs_shape"] == (3, 5)
    assert out["n_merged"] == 3
    assert out["accounting_pass_ratio"] == pytest.approx(2 / 3)
    assert out["direction_consistency_ratio"] == pytest.approx(2 / 3)


def test_run_quickcheck_from_dir():
    bs, is_, cf = _build_frames()
    tmp_path = Path(__file__).resolve().parent / f"_tmp_ch01_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=False)
    try:
        bs.to_csv(tmp_path / "balance_sheet.csv", index=False)
        is_.to_csv(tmp_path / "income_statement.csv", index=False)
        cf.to_csv(tmp_path / "cash_flow.csv", index=False)
        out = run_quickcheck_from_dir(tmp_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
    assert out["bs_shape"] == (3, 5)
    assert out["is_shape"] == (3, 3)
    assert out["cf_shape"] == (3, 3)
