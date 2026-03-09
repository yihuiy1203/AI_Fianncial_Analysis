from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from ifa.data.cleaner import run_cleaner
from ifa.data.crawler import run_crawler
from ifa.data.loader import load


def main() -> None:
    base = Path(__file__).resolve().parent / f"_tmp_ch02_demo_{uuid.uuid4().hex}"
    raw_dir = base / "raw"
    cleaned_dir = base / "cleaned"
    base.mkdir(parents=True, exist_ok=False)
    try:
        crawl_result = run_crawler(
            codes=["000001", "000002", "000063", "600519"],
            years=[2019, 2020, 2021],
            output_dir=raw_dir,
        )
        clean_result = run_cleaner(raw_dir, cleaned_dir)
        sample = load("000001", 2019, 2021, cleaned_dir=cleaned_dir)
        print(
            json.dumps(
                {
                    "crawl": crawl_result,
                    "clean_output_rows": clean_result["output_rows"],
                    "sample_shape": list(sample.shape),
                    "sample_columns": list(sample.columns),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
