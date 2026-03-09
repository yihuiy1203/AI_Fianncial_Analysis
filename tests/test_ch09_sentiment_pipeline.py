from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pandas as pd

from ifa.sentiment import (
    aggregate_weekly,
    build_sentiment_factor,
    clean_posts,
    crawl_guba,
    fit_topics,
    get_topic_keywords,
    plot_wordcloud,
    score_sentiment,
)


def test_ch09_pipeline_end_to_end():
    raw = crawl_guba("000001", "2023-01-01", "2023-01-15")
    clean = clean_posts(raw)
    assert not clean.empty

    topics, probs, model = fit_topics(clean["body"].tolist())
    assert len(topics) == len(clean)
    assert len(probs) == len(clean)

    kws = get_topic_keywords(model, top_n=5)
    assert isinstance(kws, dict)

    clean = clean.copy()
    clean["topic_id"] = topics
    clean["topic_prob"] = probs
    clean["sentiment_score"] = score_sentiment(clean["body"].tolist(), method="lexicon")

    weekly = aggregate_weekly(clean, weight_col=None)
    assert {"stock_code", "week", "sentiment_index", "post_count"}.issubset(weekly.columns)

    factor = build_sentiment_factor(clean, weight_mode="read_count")
    assert {"stock_code", "week", "sentiment_index", "sentiment_factor", "post_count"}.issubset(factor.columns)

    fig = plot_wordcloud(model, topic_id=0)
    assert fig is not None
    fig.clf()


def test_ch09_boundary_cleaning_and_neutral_score():
    df = pd.DataFrame(
        {
            "stock_code": ["000001", "000001", "000001"],
            "title": ["a", "b", "c"],
            "body": ["点击开户链接", "中性讨论", " "],
            "post_time": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "read_count": [1, 2, 3],
        }
    )

    clean = clean_posts(df, min_len=2)
    assert len(clean) == 1
    score = score_sentiment(clean["body"].tolist())[0]
    assert score == 0.0


def test_ch09_failure_invalid_inputs():
    try:
        crawl_guba("000001", "2023-01-10", "2023-01-01")
        raised_range = False
    except ValueError:
        raised_range = True
    assert raised_range

    try:
        score_sentiment(["test"], method="model")
        raised_method = False
    except ValueError:
        raised_method = True
    assert raised_method

    try:
        aggregate_weekly(pd.DataFrame({"x": [1]}))
        raised_agg = False
    except ValueError:
        raised_agg = True
    assert raised_agg
