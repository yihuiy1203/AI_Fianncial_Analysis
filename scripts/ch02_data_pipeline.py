from __future__ import annotations

import json
from pathlib import Path

from ifa.data.cleaner import run_cleaner
from ifa.data.crawler import run_crawler
from ifa.data.loader import load


def main() -> None:
    codes = ["000001", "000002", "000063"]
    years = [2019, 2020, 2021]
    raw_dir = Path("data/raw")
    cleaned_dir = Path("data/cleaned")

    crawl_result = run_crawler(codes=codes, years=years, output_dir=raw_dir)
    clean_result = run_cleaner(input_dir=raw_dir, output_dir=cleaned_dir)
    sample = load(stock_code="000001", start_year=2019, end_year=2021, cleaned_dir=cleaned_dir)

    payload = {
        "crawl": crawl_result,
        "clean": clean_result,
        "sample_rows": int(len(sample)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
