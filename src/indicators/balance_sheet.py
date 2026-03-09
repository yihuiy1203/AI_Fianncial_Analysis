from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ifa.data.loader import load


REQ_LIQ_COLS = ["current_assets", "inventory", "current_liabilities"]
REQ_LEV_COLS = ["total_assets", "total_liabilities", "total_equity"]
REQ_STR_COLS = ["current_assets", "non_current_assets", "total_assets"]


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def calc_liquidity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, REQ_LIQ_COLS)
    out = df.copy()
    for col in REQ_LIQ_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    denom = out["current_liabilities"].replace(0, pd.NA)
    out["current_ratio"] = out["current_assets"] / denom
    out["quick_ratio"] = (out["current_assets"] - out["inventory"]) / denom
    return out


def calc_leverage_ratios(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, REQ_LEV_COLS)
    out = df.copy()
    for col in REQ_LEV_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    denom = out["total_assets"].replace(0, pd.NA)
    out["debt_ratio"] = out["total_liabilities"] / denom
    out["equity_ratio"] = out["total_equity"] / denom
    return out


def calc_asset_structure(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, REQ_STR_COLS)
    out = df.copy()
    for col in REQ_STR_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    denom = out["total_assets"].replace(0, pd.NA)
    out["current_assets_share"] = out["current_assets"] / denom
    out["non_current_assets_share"] = out["non_current_assets"] / denom
    return out


REGISTRY: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "liquidity": calc_liquidity_ratios,
    "leverage": calc_leverage_ratios,
    "structure": calc_asset_structure,
}


def build_panel_with_registry(df: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    use_keys = keys or ["liquidity", "leverage", "structure"]
    for key in use_keys:
        if key not in REGISTRY:
            raise ValueError(f"Unknown indicator group: {key}")
        out = REGISTRY[key](out)
    return out


def build_balance_sheet_panel(df: pd.DataFrame) -> pd.DataFrame:
    return build_panel_with_registry(df)


def quality_summary(panel: pd.DataFrame) -> dict[str, Any]:
    num = panel.select_dtypes("number")
    return {
        "missing_rate": panel.isna().mean().to_dict(),
        "extreme_count": (num.abs() > 100).sum().to_dict(),
        "structure_share_max_deviation": float(
            (
                (
                    panel.get("current_assets_share", pd.Series(dtype=float))
                    + panel.get("non_current_assets_share", pd.Series(dtype=float))
                    - 1.0
                )
                .abs()
                .dropna()
                .max()
            )
            if "current_assets_share" in panel.columns and "non_current_assets_share" in panel.columns
            else float("nan")
        ),
    }


def run_balance_sheet_pipeline(
    code: str,
    start_year: int,
    end_year: int,
    output_path: Path | None = None,
    cleaned_dir: Path | None = None,
) -> pd.DataFrame:
    raw = load(code, start_year, end_year, cleaned_dir=cleaned_dir)
    panel = build_balance_sheet_panel(raw)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_path, index=False)
    return panel


def plot_structure(panel: pd.DataFrame, code: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_structure requires matplotlib.") from exc

    need = ["year", "current_assets_share", "non_current_assets_share"]
    _require_cols(panel, need)
    df = panel.copy().sort_values("year")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["year"], df["current_assets_share"], marker="o", label="Current Assets Share")
    ax.plot(df["year"], df["non_current_assets_share"], marker="o", label="Non-current Assets Share")
    ax.set_title(f"Asset Structure Trend - {code}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1.05)
    ax.legend()
    return fig, ax


def panel_to_json_summary(panel: pd.DataFrame) -> str:
    payload = {
        "rows": int(len(panel)),
        "columns": list(panel.columns),
        "quality": quality_summary(panel),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
