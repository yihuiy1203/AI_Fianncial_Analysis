from __future__ import annotations

import numpy as np

from .embedder import embed_texts


def retrieve(query: str, embedder, store, top_k: int = 5, threshold: float = 0.3, filters: dict | None = None) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    qvec = embed_texts([query], model=embedder, normalize=True)[0]
    raw = store.search(query_vector=qvec, top_k=top_k * 3, filters=filters)
    ranked = sorted(raw, key=lambda x: x["score"], reverse=True)

    dedup: list[dict] = []
    seen = set()
    for r in ranked:
        key = (r["metadata"].get("stock_code"), r["document"][:80])
        if key in seen:
            continue
        seen.add(key)
        if float(r["score"]) >= threshold:
            dedup.append(r)
        if len(dedup) >= top_k:
            break
    return dedup


def retrieve_with_filter(query: str, embedder, store, top_k: int = 5, threshold: float = 0.3, **filters) -> list[dict]:
    return retrieve(query, embedder, store, top_k=top_k, threshold=threshold, filters=filters or None)


def rerank(results: list[dict], max_per_stock: int = 2) -> list[dict]:
    if max_per_stock <= 0:
        raise ValueError("max_per_stock must be positive")

    out: list[dict] = []
    cnt: dict[str, int] = {}
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        code = str(r["metadata"].get("stock_code", ""))
        cnt[code] = cnt.get(code, 0) + 1
        if cnt[code] <= max_per_stock:
            out.append(r)
    return out


def eval_recall(queries: list[str], relevant_ids: list[set[str]], embedder, store, k_list: list[int] | None = None) -> dict[str, float]:
    if k_list is None:
        k_list = [1, 3, 5]
    if len(queries) != len(relevant_ids):
        raise ValueError("queries and relevant_ids length mismatch")

    out: dict[str, float] = {}
    for k in k_list:
        hit = 0
        total = len(queries)
        for q, rel in zip(queries, relevant_ids, strict=False):
            res = retrieve(q, embedder, store, top_k=k, threshold=-1.0)
            got = {r["id"] for r in res}
            if rel & got:
                hit += 1
        out[f"recall@{k}"] = hit / max(1, total)
    return out
