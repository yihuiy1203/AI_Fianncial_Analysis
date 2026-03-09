from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


TOPIC_KEYWORDS = {
    0: ["业绩", "增长", "超预期", "订单", "盈利"],
    1: ["风险", "下滑", "亏损", "压力", "套牢"],
    2: ["审计", "内控", "治理", "合规", "披露"],
}


@dataclass
class SimpleTopicModel:
    topic_keywords: dict[int, list[str]]
    topic_word_freq: dict[int, Counter]


def _tokenize(text: str) -> list[str]:
    # Keep Chinese chunks and alnum words.
    tokens = re.findall(r"[\u4e00-\u9fa5]{1,}|[A-Za-z0-9_]+", text)
    return [t.lower() for t in tokens if t.strip()]


def fit_topics(texts: list[str]) -> tuple[list[int], list[float], SimpleTopicModel]:
    topics: list[int] = []
    probs: list[float] = []
    topic_word_freq: dict[int, Counter] = defaultdict(Counter)

    for text in texts:
        text = str(text)
        best_topic = -1
        best_score = 0
        for topic_id, kws in TOPIC_KEYWORDS.items():
            score = sum(1 for kw in kws if kw in text)
            if score > best_score:
                best_score = score
                best_topic = topic_id

        if best_topic == -1:
            topics.append(-1)
            probs.append(0.0)
        else:
            topics.append(best_topic)
            probs.append(min(1.0, 0.4 + 0.2 * best_score))
            for tok in _tokenize(text):
                topic_word_freq[best_topic][tok] += 1

    model = SimpleTopicModel(topic_keywords=TOPIC_KEYWORDS, topic_word_freq=dict(topic_word_freq))
    return topics, probs, model


def get_topic_keywords(model: SimpleTopicModel, top_n: int = 10) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for tid, freq in model.topic_word_freq.items():
        out[tid] = [w for w, _ in freq.most_common(top_n)]
    return out


def plot_wordcloud(model: SimpleTopicModel, topic_id: int):
    if plt is None:
        raise ImportError("matplotlib is required for plotting")

    freq = model.topic_word_freq.get(topic_id, Counter())
    if not freq:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"No words for topic {topic_id}", ha="center", va="center")
        ax.axis("off")
        return fig

    top = freq.most_common(12)
    words = [w for w, _ in top]
    vals = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(words, vals)
    ax.set_title(f"Topic {topic_id} Keywords")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig
