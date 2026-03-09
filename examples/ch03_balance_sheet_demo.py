from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from ifa.data.cleaner import run_cleaner
from ifa.data.crawler import run_crawler
from ifa.indicators.balance_sheet import panel_to_json_summary, run_balance_sheet_pipeline


def main() -> None:
    base = Path(__file__).resolve().parent / f"_tmp_ch03_demo_{uuid.uuid4().hex}"
    raw_dir = base / "raw"
    cleaned_dir = base / "cleaned"
    feature_path = base / "features" / "balance_sheet_panel.csv"
    base.mkdir(parents=True, exist_ok=False)
    try:
        crawl = run_crawler(
            codes=["000001", "000002", "000063"],
            years=[2019, 2020, 2021],
            output_dir=raw_dir,
        )
        clean = run_cleaner(raw_dir, cleaned_dir)
        panel = run_balance_sheet_pipeline(
            code="000001",
            start_year=2019,
            end_year=2021,
            output_path=feature_path,
            cleaned_dir=cleaned_dir,
        )
        print(
            json.dumps(
                {
                    "crawl_rows": {
                        "balance_sheet": crawl["balance_sheet_rows"],
                        "income_statement": crawl["income_statement_rows"],
                        "cash_flow": crawl["cash_flow_rows"],
                    },
                    "clean_rows": clean["output_rows"],
                    "panel_summary": json.loads(panel_to_json_summary(panel)),
                    "output_file": str(feature_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
