from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from ifa.knowledge import (
    add_documents,
    create_store,
    embed_texts,
    eval_recall,
    get_stats,
    load,
    load_model,
    retrieve,
    retrieve_with_filter,
    save,
)


def build_corpus():
    texts = [
        "公司在年度报告中提示上游供货不稳定，供应链风险上升。",
        "本期应收账款坏账准备增加，回款压力加大。",
        "研发费用率提升，管理层强调技术投入与产品迭代。",
        "审计报告关键事项涉及收入确认与应收款项可回收性。",
        "股吧讨论集中在订单增长与盈利修复预期。",
    ]
    metas = [
        {"stock_code": "000001", "year": 2023, "source": "mda"},
        {"stock_code": "000001", "year": 2023, "source": "mda"},
        {"stock_code": "000002", "year": 2023, "source": "mda"},
        {"stock_code": "000002", "year": 2022, "source": "audit"},
        {"stock_code": "000063", "year": 2023, "source": "sentiment"},
    ]
    ids = [f"doc_{i}" for i in range(len(texts))]
    return texts, metas, ids


def main() -> None:
    base = Path(__file__).resolve().parent / f"_tmp_ch13_demo_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=False)
    try:
        texts, metas, ids = build_corpus()

        embedder = load_model(dim=128)
        vectors, log = embed_texts(texts, model=embedder, return_log=True)

        store = create_store(persist_dir=base)
        add_documents(store, vectors, metas, texts, ids=ids)
        stats_before = get_stats(store)

        q1 = retrieve("供应链风险", embedder, store, top_k=3, threshold=-1.0)
        q2 = retrieve_with_filter("应收账款可回收性", embedder, store, top_k=3, threshold=-1.0, source="audit")

        save(store)
        loaded = load(base)
        stats_after = get_stats(loaded)

        recall = eval_recall(
            queries=["供应链", "研发投入"],
            relevant_ids=[{"doc_0"}, {"doc_2"}],
            embedder=embedder,
            store=loaded,
            k_list=[1, 3],
        )

        print(
            json.dumps(
                {
                    "embed_log": log,
                    "stats_before": stats_before,
                    "stats_after": stats_after,
                    "query1_top": [{"id": r["id"], "score": r["score"]} for r in q1],
                    "query2_top": [{"id": r["id"], "score": r["score"]} for r in q2],
                    "recall": recall,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
