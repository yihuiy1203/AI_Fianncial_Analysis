from __future__ import annotations

import numpy as np
import pandas as pd

POS_WORDS = {"看好", "增长", "超预期", "改善", "稳健", "利好", "盈利"}
NEG_WORDS = {"风险", "下滑", "亏损", "担忧", "套牢", "利空", "压力"}


def score_sentiment(texts: list[str], method: str = "lexicon") -> list[float]:
    if method != "lexicon":
        raise ValueError("only method='lexicon' is supported in this teaching implementation")

    scores: list[float] = []
    for t in texts:
        text = str(t)
        pos = sum(1 for w in POS_WORDS if w in text)
        neg = sum(1 for w in NEG_WORDS if w in text)
        denom = max(1, pos + neg)
        scores.append((pos - neg) / denom)
    return scores


def aggregate_weekly(scores_df: pd.DataFrame, weight_col: str | None = None) -> pd.DataFrame:
    required = {"stock_code", "post_time", "sentiment_score"}
    missing = sorted(required - set(scores_df.columns))
    if missing:
        raise ValueError(f"scores_df missing required columns: {missing}")

    out = scores_df.copy()
    out["post_time"] = pd.to_datetime(out["post_time"], errors="coerce")
    out = out.dropna(subset=["post_time"])
    out["week"] = out["post_time"].dt.to_period("W-MON").dt.start_time.dt.strftime("%Y-%m-%d")

    if weight_col is None:
        grp = out.groupby(["stock_code", "week"], as_index=False).agg(
            sentiment_index=("sentiment_score", "mean"),
            post_count=("sentiment_score", "size"),
        )
        return grp

    if weight_col not in out.columns:
        raise ValueError(f"weight_col not in scores_df: {weight_col}")

    out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0)

    def _wavg(g: pd.DataFrame) -> float:
        w = g[weight_col].to_numpy(dtype=float)
        x = g["sentiment_score"].to_numpy(dtype=float)
        if np.all(w == 0):
            return float(np.mean(x))
        return float(np.sum(x * w) / np.sum(w))

    grp = (
        out.groupby(["stock_code", "week"], as_index=False)
        .apply(lambda g: pd.Series({"sentiment_index": _wavg(g), "post_count": len(g)}), include_groups=False)
    )
    return grp


def build_sentiment_factor(scores_df: pd.DataFrame, weight_mode: str = "equal") -> pd.DataFrame:
    if weight_mode not in {"equal", "read_count"}:
        raise ValueError("weight_mode must be 'equal' or 'read_count'")

    weekly = aggregate_weekly(scores_df, weight_col=None if weight_mode == "equal" else "read_count")
    if weekly.empty:
        weekly["sentiment_factor"] = []
        return weekly

    weekly = weekly.copy()
    weekly["sentiment_factor"] = weekly.groupby("stock_code")["sentiment_index"].transform(
        lambda s: 0.0 if s.std(ddof=0) == 0 else (s - s.mean()) / s.std(ddof=0)
    )
    return weekly[["stock_code", "week", "sentiment_index", "sentiment_factor", "post_count"]]
