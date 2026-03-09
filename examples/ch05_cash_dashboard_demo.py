from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from ifa.data.cleaner import run_cleaner
from ifa.data.crawler import run_crawler
from ifa.indicators.dashboard import build_full_panel, export_to_excel


def main() -> None:
    base = Path(__file__).resolve().parent / f"_tmp_ch05_demo_{uuid.uuid4().hex}"
    raw_dir = base / "raw"
    cleaned_dir = base / "cleaned"
    xlsx_path = base / "features" / "ch05_dashboard.xlsx"
    base.mkdir(parents=True, exist_ok=False)
    try:
        crawl = run_crawler(
            codes=["000001", "000002", "000063"],
            years=[2019, 2020, 2021, 2022],
            output_dir=raw_dir,
        )
        clean = run_cleaner(raw_dir, cleaned_dir)
        panel = build_full_panel("000001", 2019, 2022, cleaned_dir=cleaned_dir)
        export_to_excel(panel, xlsx_path)
        print(
            json.dumps(
                {
                    "crawl_rows": {
                        "balance_sheet": crawl["balance_sheet_rows"],
                        "income_statement": crawl["income_statement_rows"],
                        "cash_flow": crawl["cash_flow_rows"],
                    },
                    "clean_rows": clean["output_rows"],
                    "panel_shape": list(panel.shape),
                    "panel_columns_sample": panel.columns[:12].tolist(),
                    "excel_output": str(xlsx_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
