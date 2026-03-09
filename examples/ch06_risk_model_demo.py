from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from ifa.data.cleaner import run_cleaner
from ifa.data.crawler import run_crawler
from ifa.indicators.dashboard import build_full_panel
from ifa.risk.pipeline import run_pipeline


def main() -> None:
    base = Path(__file__).resolve().parent / f"_tmp_ch06_demo_{uuid.uuid4().hex}"
    raw_dir = base / "raw"
    cleaned_dir = base / "cleaned"
    base.mkdir(parents=True, exist_ok=False)
    try:
        run_crawler(codes=["000001", "000002", "000063", "600519"], years=[2019, 2020, 2021, 2022], output_dir=raw_dir)
        run_cleaner(raw_dir, cleaned_dir)
        panel = pd.concat(
            [
                build_full_panel("000001", 2019, 2022, cleaned_dir=cleaned_dir),
                build_full_panel("000002", 2019, 2022, cleaned_dir=cleaned_dir),
                build_full_panel("000063", 2019, 2022, cleaned_dir=cleaned_dir),
                build_full_panel("600519", 2019, 2022, cleaned_dir=cleaned_dir),
            ],
            ignore_index=True,
        )
        labels = panel[["stock_code", "year"]].copy()
        labels["is_st"] = (
            (panel["debt_ratio"] > panel["debt_ratio"].median())
            & (panel["cash_earnings_ratio"] < panel["cash_earnings_ratio"].median())
        ).astype(int)

        _, metrics, sweep, importance, out = run_pipeline(panel, labels, train_year_end=2021, threshold=0.45)
        print(
            json.dumps(
                {
                    "panel_shape": list(panel.shape),
                    "metrics": {
                        "auc": metrics["auc"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                    },
                    "threshold_rows": int(len(sweep)),
                    "top_features": importance.head(5).to_dict(),
                    "scored_rows": int(len(out)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
