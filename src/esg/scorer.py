from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    dim: str
    direction: str


def _validate_specs(specs: Iterable[IndicatorSpec]) -> list[IndicatorSpec]:
    out = list(specs)
    if not out:
        raise ValueError("specs must not be empty")
    for s in out:
        if s.dim not in {"E", "S", "G"}:
            raise ValueError(f"invalid dim for {s.name}: {s.dim}")
        if s.direction not in {"positive", "negative"}:
            raise ValueError(f"invalid direction for {s.name}: {s.direction}")
    return out


def set_weights(
    specs: Iterable[IndicatorSpec],
    dim_weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    use_specs = _validate_specs(specs)

    if dim_weights is None:
        dim_weights = {"E": 1.0 / 3, "S": 1.0 / 3, "G": 1.0 / 3}
    if set(dim_weights.keys()) != {"E", "S", "G"}:
        raise ValueError("dim_weights must contain E/S/G keys")

    total = float(sum(dim_weights.values()))
    if total <= 0:
        raise ValueError("sum(dim_weights) must be positive")
    norm_dim_weights = {k: float(v) / total for k, v in dim_weights.items()}

    ind_weights: dict[str, float] = {}
    for dim in ["E", "S", "G"]:
        cols = [s.name for s in use_specs if s.dim == dim]
        if not cols:
            raise ValueError(f"no indicators found for dim {dim}")
        weight = 1.0 / len(cols)
        for col in cols:
            ind_weights[col] = weight

    return norm_dim_weights, ind_weights


def _minmax(series: pd.Series) -> pd.Series:
    lo = series.min(skipna=True)
    hi = series.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - lo) / (hi - lo)


def normalize_indicators(df: pd.DataFrame, specs: Iterable[IndicatorSpec]) -> pd.DataFrame:
    use_specs = _validate_specs(specs)
    out = df.copy()

    required = {"stock_code", "year", *[s.name for s in use_specs]}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"df missing required columns: {missing}")

    for s in use_specs:
        x = pd.to_numeric(out[s.name], errors="coerce")
        if s.direction == "negative":
            x = -x
        out[f"{s.name}_norm"] = _minmax(x)

    return out


def compute_scores(
    df: pd.DataFrame,
    specs: Iterable[IndicatorSpec],
    dim_weights: dict[str, float],
    ind_weights: dict[str, float],
) -> pd.DataFrame:
    use_specs = _validate_specs(specs)
    out = df.copy()

    for dim in ["E", "S", "G"]:
        specs_dim = [s for s in use_specs if s.dim == dim]
        norm_cols = [f"{s.name}_norm" for s in specs_dim]
        missing = [c for c in norm_cols if c not in out.columns]
        if missing:
            raise ValueError(f"normalized columns missing for dim {dim}: {missing}")
        w = np.array([float(ind_weights[s.name]) for s in specs_dim], dtype=float)
        out[f"{dim}_score"] = out[norm_cols].to_numpy() @ w

    out["total_score"] = (
        float(dim_weights["E"]) * out["E_score"]
        + float(dim_weights["S"]) * out["S_score"]
        + float(dim_weights["G"]) * out["G_score"]
    )

    result_cols = ["stock_code", "year", "E_score", "S_score", "G_score", "total_score"]
    return out[result_cols].copy()
