from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    **kwargs: Any,
):
    # Preferred path for chapter parity; fallback keeps the pipeline runnable
    # when xgboost is not installed in the environment.
    try:
        from xgboost import XGBClassifier  # type: ignore

        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale = float(neg / max(pos, 1))
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale,
            random_state=random_state,
            **kwargs,
        )
        model.fit(X_train, y_train)
        return model
    except Exception:
        model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        return model


def predict_risk_score(model, X: pd.DataFrame, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        # HistGradientBoosting exposes decision_function in some versions.
        raw = model.decision_function(X)
        prob = 1.0 / (1.0 + np.exp(-raw))
    pred = (prob >= threshold).astype(int)
    return prob, pred


def get_feature_importance(model, feature_names: list[str]) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:
        # fallback for estimators without native importance
        imp = np.zeros(len(feature_names), dtype=float)
    s = pd.Series(imp, index=feature_names)
    return s.sort_values(ascending=False)


def save_model(model, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
