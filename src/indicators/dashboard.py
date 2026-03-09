from __future__ import annotations

from pathlib import Path

import pandas as pd

from ifa.config import settings
from ifa.indicators.balance_sheet import build_balance_sheet_panel
from ifa.indicators.cash_flow import calc_cash_earnings_ratio, calc_cash_flow_structure, calc_fcf
from ifa.indicators.dupont import build_dupont_features
from ifa.indicators.income_statement import build_income_features


def _normalize_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stock_code"] = out["stock_code"].astype(str).str.strip().str.zfill(6)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def _merge_all(frames: list[pd.DataFrame], on: list[str]) -> pd.DataFrame:
    it = iter(frames)
    merged = next(it)
    for f in it:
        merged = merged.merge(f, on=on, how="inner")
    return merged


def _pick(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()


def build_full_panel(
    code: str,
    start_year: int,
    end_year: int,
    cleaned_dir: Path | None = None,
) -> pd.DataFrame:
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year.")

    cdir = cleaned_dir or settings.get_path("data_cleaned")
    bs = _normalize_key(pd.read_csv(cdir / "balance_sheet.csv"))
    inc = _normalize_key(pd.read_csv(cdir / "income_statement.csv"))
    cf = _normalize_key(pd.read_csv(cdir / "cash_flow.csv"))

    code_norm = str(code).strip().zfill(6)
    for name, df in [("bs", bs), ("inc", inc), ("cf", cf)]:
        if "stock_code" not in df.columns or "year" not in df.columns:
            raise ValueError(f"{name} missing key columns.")

    bs = bs[(bs["stock_code"] == code_norm) & (bs["year"].between(start_year, end_year))]
    inc = inc[(inc["stock_code"] == code_norm) & (inc["year"].between(start_year, end_year))]
    cf = cf[(cf["stock_code"] == code_norm) & (cf["year"].between(start_year, end_year))]

    base = _merge_all(
        [
            bs,
            inc,
            cf,
        ],
        on=["stock_code", "year"],
    )

    bs_panel = build_balance_sheet_panel(base)
    income_in = base.assign(avg_total_assets=base["total_assets"], avg_equity=base["total_equity"])
    inc_panel = build_income_features(income_in)
    dup_panel = build_dupont_features(inc, bs)
    cf_structure = calc_cash_flow_structure(base)
    fcf_panel = calc_fcf(base)
    cer_panel = calc_cash_earnings_ratio(base)

    panel = _merge_all(
        [
            _pick(bs_panel, ["stock_code", "year", "current_ratio", "quick_ratio", "debt_ratio", "equity_ratio"]),
            _pick(
                inc_panel,
                [
                    "stock_code",
                    "year",
                    "gross_margin",
                    "net_margin",
                    "roe",
                    "roa",
                    "core_profit_ratio",
                    "non_recurring_ratio",
                    "revenue_yoy",
                    "profit_yoy",
                ],
            ),
            _pick(
                dup_panel,
                [
                    "stock_code",
                    "year",
                    "tax_burden",
                    "interest_burden",
                    "operating_margin",
                    "asset_turnover",
                    "equity_multiplier",
                    "roe_dupont",
                    "roe_diff",
                ],
            ),
            _pick(
                cf_structure,
                ["stock_code", "year", "operating_cf_share", "investing_cf_share", "financing_cf_share"],
            ),
            _pick(fcf_panel, ["stock_code", "year", "fcf"]),
            _pick(cer_panel, ["stock_code", "year", "cash_earnings_ratio"]),
        ],
        on=["stock_code", "year"],
    )
    return panel.sort_values(["stock_code", "year"]).reset_index(drop=True)


def export_to_excel(panel: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        base_cols = ["stock_code", "year"]
        solvency = base_cols + [c for c in ["current_ratio", "quick_ratio", "debt_ratio", "equity_ratio"] if c in panel]
        profitability = base_cols + [
            c
            for c in [
                "gross_margin",
                "net_margin",
                "roe",
                "roa",
                "core_profit_ratio",
                "non_recurring_ratio",
                "revenue_yoy",
                "profit_yoy",
                "roe_dupont",
                "roe_diff",
            ]
            if c in panel
        ]
        cash = base_cols + [
            c
            for c in [
                "operating_cf_share",
                "investing_cf_share",
                "financing_cf_share",
                "fcf",
                "cash_earnings_ratio",
            ]
            if c in panel
        ]
        panel[solvency].to_excel(writer, sheet_name="solvency", index=False)
        panel[profitability].to_excel(writer, sheet_name="profitability", index=False)
        panel[cash].to_excel(writer, sheet_name="cash_flow", index=False)
    return output_path


def plot_dashboard(panel: pd.DataFrame, code: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_dashboard requires matplotlib.") from exc

    sub = panel[panel["stock_code"] == str(code).strip().zfill(6)].sort_values("year")
    if sub.empty:
        raise ValueError("No rows available for plotting.")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(sub["year"], sub["current_ratio"], marker="o", label="Current Ratio")
    axes[0].plot(sub["year"], sub["debt_ratio"], marker="o", label="Debt Ratio")
    axes[0].set_title("Solvency")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(sub["year"], sub["roe"], marker="o", label="ROE")
    axes[1].plot(sub["year"], sub["net_margin"], marker="o", label="Net Margin")
    axes[1].set_title("Profitability")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(sub["year"], sub["fcf"], marker="o", label="FCF")
    axes[2].plot(sub["year"], sub["cash_earnings_ratio"], marker="o", label="Cash Earnings Ratio")
    axes[2].set_title("Cash Flow")
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    axes[2].set_xlabel("Year")

    fig.suptitle(f"Financial Dashboard - {str(code).strip().zfill(6)}")
    fig.tight_layout()
    return fig, axes
