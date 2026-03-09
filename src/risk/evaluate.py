from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve


def calc_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"auc": float(auc), "precision": float(p), "recall": float(r), "f1": float(f1), "cm": cm}


def threshold_sweep(y_true, y_prob, grid: list[float] | None = None) -> pd.DataFrame:
    if grid is None:
        grid = [i / 100 for i in range(20, 81, 5)]
    rows = []
    for th in grid:
        m = calc_metrics(y_true, y_prob, threshold=th)
        rows.append(
            {
                "threshold": th,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "auc": m["auc"],
            }
        )
    return pd.DataFrame(rows)


def roc_points(y_true, y_prob) -> pd.DataFrame:
    if len(set(y_true)) <= 1:
        return pd.DataFrame({"fpr": [], "tpr": [], "thresholds": []})
    fpr, tpr, th = roc_curve(y_true, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": th})
