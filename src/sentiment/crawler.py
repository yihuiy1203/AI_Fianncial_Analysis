from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def _to_dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def crawl_guba(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Teaching/demo crawler: generate deterministic public-discussion samples."""
    start = _to_dt(start_date)
    end = _to_dt(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")

    templates = [
        ("业绩预期", "公司订单增长，业绩可能超预期，继续看好。", 1200),
        ("风险提示", "回款压力上升，担忧利润率下滑，注意风险。", 900),
        ("治理讨论", "审计意见和内控改进值得跟踪，偏中性。", 600),
    ]

    rows: list[dict[str, object]] = []
    i = 0
    cur = start
    while cur <= end:
        title, body, base_read = templates[i % len(templates)]
        rows.append(
            {
                "stock_code": str(stock_code).zfill(6),
                "title": f"[{stock_code}] {title}",
                "body": body,
                "post_time": cur.strftime("%Y-%m-%d"),
                "read_count": base_read + (i % 5) * 40,
            }
        )
        # Add one noisy short/ad-like post every 5 rows for cleaning tests.
        if i % 5 == 0:
            rows.append(
                {
                    "stock_code": str(stock_code).zfill(6),
                    "title": f"[{stock_code}] 广告",
                    "body": "点击开户链接",
                    "post_time": cur.strftime("%Y-%m-%d"),
                    "read_count": 20,
                }
            )
        i += 1
        cur += timedelta(days=1)

    return pd.DataFrame(rows)


def clean_posts(posts_df: pd.DataFrame, min_len: int = 8) -> pd.DataFrame:
    required = {"stock_code", "title", "body", "post_time", "read_count"}
    missing = sorted(required - set(posts_df.columns))
    if missing:
        raise ValueError(f"posts_df missing required columns: {missing}")

    out = posts_df.copy()
    out["body"] = out["body"].astype(str).str.strip()
    out = out[~out["body"].str.contains("广告|开户链接|加群", regex=True)]
    out = out[out["body"].str.len() >= min_len]
    out["post_time"] = pd.to_datetime(out["post_time"], errors="coerce")
    out["read_count"] = pd.to_numeric(out["read_count"], errors="coerce").fillna(0)
    out = out.dropna(subset=["post_time"]).drop_duplicates(subset=["stock_code", "title", "body", "post_time"])
    out = out.sort_values("post_time").reset_index(drop=True)
    out["post_time"] = out["post_time"].dt.strftime("%Y-%m-%d")
    return out
