from __future__ import annotations

import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency guard
    plt = None


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting functions")


def plot_radar(scores_df: pd.DataFrame, code: str, year: int):
    _require_matplotlib()
    matched = scores_df[(scores_df["stock_code"].astype(str) == str(code)) & (scores_df["year"] == year)]
    if matched.empty:
        raise ValueError(f"no row found for stock_code={code}, year={year}")
    row = matched.iloc[0]

    labels = ["E", "S", "G"]
    values = [float(row["E_score"]), float(row["S_score"]), float(row["G_score"])]
    angles = [0.0, 2.0943951023931953, 4.1887902047863905]

    values = values + values[:1]
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"ESG Radar {code} {year}")
    return fig


def plot_comparison_panel(scores_df: pd.DataFrame, year: int, top_n: int = 10):
    _require_matplotlib()
    sub = scores_df[scores_df["year"] == year].nlargest(top_n, "total_score")
    if sub.empty:
        raise ValueError(f"no rows found for year={year}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(sub["stock_code"].astype(str), sub["total_score"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("total_score")
    ax.set_title(f"Top {top_n} ESG Scores ({year})")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_esg_vs_risk(esg_df: pd.DataFrame, risk_df: pd.DataFrame, year: int):
    _require_matplotlib()
    required_esg = {"stock_code", "year", "total_score"}
    required_risk = {"stock_code", "year", "risk_score"}
    missing_esg = sorted(required_esg - set(esg_df.columns))
    missing_risk = sorted(required_risk - set(risk_df.columns))
    if missing_esg:
        raise ValueError(f"esg_df missing required columns: {missing_esg}")
    if missing_risk:
        raise ValueError(f"risk_df missing required columns: {missing_risk}")

    merged = esg_df.merge(risk_df[["stock_code", "year", "risk_score"]], on=["stock_code", "year"], how="inner")
    merged = merged[merged["year"] == year].copy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(merged["total_score"], merged["risk_score"], alpha=0.8)
    ax.set_xlabel("ESG total_score")
    ax.set_ylabel("risk_score")
    ax.set_title(f"ESG vs Risk ({year})")
    fig.tight_layout()
    return merged, fig
