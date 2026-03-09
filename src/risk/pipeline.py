from __future__ import annotations

from pathlib import Path

import pandas as pd

from ifa.config import settings
from ifa.risk.evaluate import calc_metrics, threshold_sweep
from ifa.risk.features import build_feature_matrix
from ifa.risk.model import get_feature_importance, predict_risk_score, save_model, train_xgboost


def run_pipeline(
    panel_df: pd.DataFrame,
    label_df: pd.DataFrame,
    train_year_end: int = 2021,
    threshold: float = 0.45,
):
    X, y, meta = build_feature_matrix(panel_df, label_df)
    train_idx = meta["year"] <= train_year_end
    test_idx = ~train_idx
    if int(train_idx.sum()) == 0 or int(test_idx.sum()) == 0:
        raise ValueError("time split produced empty train/test set; adjust train_year_end or sample window.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = train_xgboost(X_train, y_train)
    y_prob, y_pred = predict_risk_score(model, X_test, threshold=threshold)
    metrics = calc_metrics(y_test, y_prob, threshold=threshold)
    sweep = threshold_sweep(y_test, y_prob)
    importance = get_feature_importance(model, list(X.columns))

    out = meta[test_idx].copy()
    out["y_true"] = y_test.values
    out["risk_prob"] = y_prob
    out["y_pred"] = y_pred
    return model, metrics, sweep, importance, out


def save_pipeline_artifacts(model, scored_df: pd.DataFrame, model_name: str = "risk_model.pkl") -> tuple[Path, Path]:
    model_path = save_model(model, settings.get_path("outputs_models") / model_name)
    features_dir = settings.get_path("data_features")
    features_dir.mkdir(parents=True, exist_ok=True)
    score_path = features_dir / "risk_scores.csv"
    scored_df.to_csv(score_path, index=False)
    return model_path, score_path
