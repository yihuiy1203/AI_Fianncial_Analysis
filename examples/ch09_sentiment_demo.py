from __future__ import annotations

import json

from ifa.sentiment import (
    build_sentiment_factor,
    clean_posts,
    crawl_guba,
    fit_topics,
    get_topic_keywords,
    plot_wordcloud,
    score_sentiment,
)


def main() -> None:
    raw = crawl_guba("000001", "2023-01-01", "2023-01-21")
    posts = clean_posts(raw)

    topics, probs, model = fit_topics(posts["body"].tolist())
    posts = posts.copy()
    posts["topic_id"] = topics
    posts["topic_prob"] = probs
    posts["sentiment_score"] = score_sentiment(posts["body"].tolist(), method="lexicon")

    factor = build_sentiment_factor(posts, weight_mode="read_count")
    kws = get_topic_keywords(model, top_n=5)

    fig = plot_wordcloud(model, topic_id=0)
    fig.clf()

    print(
        json.dumps(
            {
                "rows_raw": int(len(raw)),
                "rows_clean": int(len(posts)),
                "topic_counts": posts["topic_id"].value_counts().to_dict(),
                "weekly_rows": int(len(factor)),
                "keywords_topic0": kws.get(0, []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
