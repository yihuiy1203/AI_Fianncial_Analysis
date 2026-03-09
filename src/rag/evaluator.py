from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EvalItem:
    query: str
    answer: str
    contexts: list[str]
    citations: list[int]


def _split_sentences(text: str) -> list[str]:
    seps = ["。", "!", "！", "?", "？", "\n"]
    out = [text]
    for s in seps:
        tmp = []
        for t in out:
            tmp.extend([x.strip() for x in t.split(s) if x.strip()])
        out = tmp
    return out


def eval_faithfulness(answer: str, contexts: list[str]) -> float:
    sents = _split_sentences(answer)
    if not sents:
        return 0.0
    if not contexts:
        return 0.0

    hit = 0
    for s in sents:
        # lightweight support check: sentence prefix overlap with any context
        token = s[:6]
        if any(token and token in c for c in contexts):
            hit += 1
    return hit / len(sents)


def eval_relevance(query: str, answer: str) -> float:
    q_terms = [x for x in _split_sentences(query) if x]
    if not q_terms:
        return 0.0
    # character-overlap relevance for Chinese queries without whitespace tokenization
    q_chars = {c for c in "".join(q_terms) if c.strip() and c not in {"？", "?", "，", ",", "。", "."}}
    a_chars = {c for c in answer if c.strip() and c not in {"？", "?", "，", ",", "。", "."}}
    if not q_chars:
        return 0.0
    return len(q_chars & a_chars) / len(q_chars)


def eval_batch(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        contexts = [d.get("document", d.get("text", "")) for d in r.get("retrieved_chunks", [])]
        rows.append(
            {
                "query": r.get("query", ""),
                "faithfulness": eval_faithfulness(r.get("answer", ""), contexts),
                "relevance": eval_relevance(r.get("query", ""), r.get("answer", "")),
                "n_context": len(contexts),
                "n_citations": len(r.get("citations", [])),
            }
        )
    return pd.DataFrame(rows)


def generate_eval_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    df = eval_batch(records)
    if df.empty:
        return {
            "n": 0,
            "faithfulness_mean": 0.0,
            "relevance_mean": 0.0,
            "citation_rate": 0.0,
            "detail": df,
        }

    citation_rate = float((df["n_citations"] > 0).mean())
    return {
        "n": int(len(df)),
        "faithfulness_mean": float(df["faithfulness"].mean()),
        "relevance_mean": float(df["relevance"].mean()),
        "citation_rate": citation_rate,
        "detail": df,
    }
