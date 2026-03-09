from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from ifa.data.ch01_quickcheck import run_quickcheck_from_dir


def _write_demo_csvs(raw_dir: Path) -> None:
    balance_sheet = pd.DataFrame(
        [
            {
                "stock_code": "000001.SZ",
                "report_date": "2024-12-31",
                "total_assets": 120.0,
                "total_liabilities": 70.0,
                "total_equity": 50.0,
            },
            {
                "stock_code": "000002.SZ",
                "report_date": "2024-12-31",
                "total_assets": 200.0,
                "total_liabilities": 140.0,
                "total_equity": 59.0,
            },
            {
                "stock_code": "000003.SZ",
                "report_date": "2024-12-31",
                "total_assets": 90.0,
                "total_liabilities": 30.0,
                "total_equity": 60.0,
            },
        ]
    )
    income_statement = pd.DataFrame(
        [
            {"stock_code": "000001.SZ", "report_date": "2024-12-31", "net_profit": 12.0},
            {"stock_code": "000002.SZ", "report_date": "2024-12-31", "net_profit": -3.0},
            {"stock_code": "000003.SZ", "report_date": "2024-12-31", "net_profit": 8.0},
        ]
    )
    cash_flow = pd.DataFrame(
        [
            {"stock_code": "000001.SZ", "report_date": "2024-12-31", "operating_cash_flow": 10.0},
            {"stock_code": "000002.SZ", "report_date": "2024-12-31", "operating_cash_flow": -1.0},
            {"stock_code": "000003.SZ", "report_date": "2024-12-31", "operating_cash_flow": -2.0},
        ]
    )
    balance_sheet.to_csv(raw_dir / "balance_sheet.csv", index=False)
    income_statement.to_csv(raw_dir / "income_statement.csv", index=False)
    cash_flow.to_csv(raw_dir / "cash_flow.csv", index=False)


def main() -> None:
    raw_dir = Path(__file__).resolve().parent / f"_tmp_ch01_demo_{uuid.uuid4().hex}"
    raw_dir.mkdir(parents=True, exist_ok=False)
    try:
        _write_demo_csvs(raw_dir)
        result = run_quickcheck_from_dir(raw_dir)
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
