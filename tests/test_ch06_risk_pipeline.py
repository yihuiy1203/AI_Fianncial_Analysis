from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd

from ifa.data.cleaner import run_cleaner
from ifa.data.crawler import run_crawler
from ifa.indicators.dashboard import build_full_panel
from ifa.risk.evaluate import calc_metrics, threshold_sweep
from ifa.risk.features import build_feature_matrix
from ifa.risk.model import get_feature_importance, predict_risk_score, train_xgboost
from ifa.risk.pipeline import run_pipeline


def _make_panel_and_labels(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = base / "raw"
    cleaned_dir = base / "cleaned"
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
    # Synthetic label for stable testing only.
    risk_flag = (
        (panel["debt_ratio"] > panel["debt_ratio"].median())
        & (panel["cash_earnings_ratio"] < panel["cash_earnings_ratio"].median())
    ).astype(int)
    label = panel[["stock_code", "year"]].copy()
    label["is_st"] = risk_flag
    return panel, label


def test_feature_matrix_and_model_training():
    base = Path(__file__).resolve().parent / f"_tmp_ch06_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=False)
    try:
        panel, label = _make_panel_and_labels(base)
        X, y, meta = build_feature_matrix(panel, label)
        assert not X.empty
        assert len(X) == len(y) == len(meta)

        model = train_xgboost(X, y)
        prob, pred = predict_risk_score(model, X, threshold=0.5)
        m = calc_metrics(y, prob, threshold=0.5)
        assert len(prob) == len(pred) == len(y)
        assert set(m.keys()) == {"auc", "precision", "recall", "f1", "cm"}

        imp = get_feature_importance(model, list(X.columns))
        assert len(imp) == X.shape[1]

        sweep = threshold_sweep(y, prob)
        assert {"threshold", "precision", "recall", "f1", "auc"}.issubset(sweep.columns)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_run_pipeline_returns_scored_output():
    base = Path(__file__).resolve().parent / f"_tmp_ch06_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=False)
    try:
        panel, label = _make_panel_and_labels(base)
        model, metrics, sweep, importance, out = run_pipeline(panel, label, train_year_end=2021, threshold=0.45)
        assert model is not None
        assert "auc" in metrics
        assert not sweep.empty
        assert not importance.empty
        assert {"stock_code", "year", "y_true", "risk_prob", "y_pred"}.issubset(out.columns)
    finally:
        shutil.rmtree(base, ignore_errors=True)
