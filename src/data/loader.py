from __future__ import annotations

from pathlib import Path

import pandas as pd

from ifa.config import settings


def _load_cleaned_table(cleaned_dir: Path, name: str) -> pd.DataFrame:
    path = cleaned_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cleaned table not found: {path}")
    df = pd.read_csv(path)
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.zfill(6)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    if "year" not in df.columns:
        df["year"] = df["report_date"].dt.year
    return df


def load(
    stock_code: str,
    start_year: int,
    end_year: int,
    cleaned_dir: Path | None = None,
) -> pd.DataFrame:
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year.")

    cdir = cleaned_dir or settings.get_path("data_cleaned")
    stock_code_norm = str(stock_code).strip().zfill(6)
    bs = _load_cleaned_table(cdir, "balance_sheet")
    is_ = _load_cleaned_table(cdir, "income_statement")
    cf = _load_cleaned_table(cdir, "cash_flow")

    key = ["stock_code", "report_date", "year"]
    panel = bs.merge(is_, on=key, how="inner").merge(cf, on=key, how="inner")
    panel = panel[(panel["stock_code"] == stock_code_norm) & (panel["year"].between(start_year, end_year))]
    panel = panel.sort_values("report_date").reset_index(drop=True)
    return panel


def load_all(cleaned_dir: Path | None = None) -> pd.DataFrame:
    cdir = cleaned_dir or settings.get_path("data_cleaned")
    bs = _load_cleaned_table(cdir, "balance_sheet")
    is_ = _load_cleaned_table(cdir, "income_statement")
    cf = _load_cleaned_table(cdir, "cash_flow")
    key = ["stock_code", "report_date", "year"]
    return bs.merge(is_, on=key, how="inner").merge(cf, on=key, how="inner").sort_values(key).reset_index(drop=True)
