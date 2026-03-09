from __future__ import annotations

import numpy as np
import pandas as pd

from ifa.causal.event_study import (
    EventStudyConfig,
    build_event_panel,
    estimate_normal_return,
    run_event_study,
)


def _make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=380)
    market_ret = rng.normal(0.0002, 0.01, size=len(dates))
    market = pd.DataFrame({"date": dates, "market_return": market_ret})

    returns_rows = []
    events_rows = []
    for i, code in enumerate(["000001", "000002", "000063"]):
        alpha = 0.0001 * (i + 1)
        beta = 1.0 + 0.1 * i
        eps = rng.normal(0.0, 0.012, size=len(dates))
        ret = alpha + beta * market_ret + eps

        event_day = dates[260 + i * 5]
        # Negative shock around event window.
        shock_idx = (dates >= event_day) & (dates <= event_day + pd.Timedelta(days=2))
        ret[shock_idx] += -0.02

        returns_rows.extend(
            {
                "stock_code": code,
                "date": d,
                "daily_return": r,
            }
            for d, r in zip(dates, ret)
        )
        events_rows.append({"event_id": f"E{i+1}", "stock_code": code, "event_date": event_day})

    returns = pd.DataFrame(returns_rows)
    events = pd.DataFrame(events_rows)
    return events, returns, market


def test_ch10_event_study_end_to_end_negative_car():
    events, returns, market = _make_data()
    cfg = EventStudyConfig(est_start=-180, est_end=-20, event_start=-3, event_end=5, min_est_obs=80)

    ar_panel, summary, stats = run_event_study(events, returns, market, cfg)
    assert not ar_panel.empty
    assert not summary.empty
    assert {"aar", "acar", "se_car", "n"}.issubset(summary.columns)
    assert stats["n"] >= 2
    assert stats["mean_car"] < 0


def test_ch10_build_event_panel_aligns_to_next_trading_day():
    events, returns, market = _make_data()
    # Use a Sunday to verify alignment logic.
    events = events.copy()
    events.loc[0, "event_date"] = pd.Timestamp("2023-01-08")

    panel = build_event_panel(events, returns, market)
    one = panel[panel["event_id"] == events.loc[0, "event_id"]]
    assert not one.empty
    t0_row = one[one["t"] == 0].iloc[0]
    assert pd.Timestamp(t0_row["event_day"]).weekday() < 5


def test_ch10_estimation_window_too_short_raises():
    events, returns, market = _make_data()
    panel = build_event_panel(events, returns, market)
    one = panel[panel["event_id"] == "E1"]
    cfg = EventStudyConfig(est_start=-20, est_end=-1, min_est_obs=50)

    try:
        estimate_normal_return(one, cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
