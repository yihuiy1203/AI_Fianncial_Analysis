from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ifa.causal.event_study import EventStudyConfig, plot_event_window, run_event_study


def build_demo_data(seed: int = 2026):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=360)
    market_ret = rng.normal(0.0003, 0.01, size=len(dates))
    market = pd.DataFrame({"date": dates, "market_return": market_ret})

    rows = []
    events = []
    for i, code in enumerate(["000001", "000002", "600519", "000063"]):
        beta = 0.9 + 0.15 * i
        alpha = 0.0001
        eps = rng.normal(0.0, 0.011, size=len(dates))
        ret = alpha + beta * market_ret + eps

        evt = dates[240 + i * 8]
        ret[(dates >= evt) & (dates <= evt + pd.Timedelta(days=2))] += -0.015

        rows.extend(
            {"stock_code": code, "date": d, "daily_return": r}
            for d, r in zip(dates, ret)
        )
        events.append({"event_id": f"Q{i+1}", "stock_code": code, "event_date": evt})

    return pd.DataFrame(events), pd.DataFrame(rows), market


def main() -> None:
    events, returns, market = build_demo_data()
    cfg = EventStudyConfig(est_start=-180, est_end=-20, event_start=-3, event_end=7, min_est_obs=90)

    ar_panel, summary, stats = run_event_study(events, returns, market, cfg)
    fig = plot_event_window(summary, cfg, title="CH10 Demo ACAR")
    fig.clf()

    print(
        json.dumps(
            {
                "events": int(events.shape[0]),
                "ar_rows": int(ar_panel.shape[0]),
                "summary_rows": int(summary.shape[0]),
                "stats": stats,
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
