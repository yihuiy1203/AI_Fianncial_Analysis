from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ifa.risk.pipeline import run_pipeline, save_pipeline_artifacts


def main() -> None:
    panel = pd.read_csv(Path("data/features/full_panel_ch05.csv"))
    labels = panel[["stock_code", "year"]].copy()
    labels["is_st"] = (
        (panel["debt_ratio"] > panel["debt_ratio"].median())
        & (panel["cash_earnings_ratio"] < panel["cash_earnings_ratio"].median())
    ).astype(int)

    model, metrics, sweep, _, out = run_pipeline(panel, labels, train_year_end=2021, threshold=0.45)
    model_path, score_path = save_pipeline_artifacts(model, out)
    sweep_path = Path("data/features/risk_threshold_sweep.csv")
    sweep.to_csv(sweep_path, index=False)

    print(
        json.dumps(
            {
                "metrics": {
                    "auc": metrics["auc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                },
                "model_path": str(model_path),
                "score_path": str(score_path),
                "sweep_path": str(sweep_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
