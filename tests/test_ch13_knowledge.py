from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np

from ifa.knowledge import (
    add_documents,
    create_store,
    embed_texts,
    eval_recall,
    get_stats,
    load,
    load_model,
    rerank,
    retrieve,
    retrieve_with_filter,
    save,
)


def _sample_docs():
    texts = [
        "公司披露上游原材料供应不稳定，存在供应链风险。",
        "本年度应收账款减值准备计提比例提高，坏账风险上升。",
        "管理层表示研发投入持续增长，研发费用率提升。",
        "审计报告提到关键审计事项为收入确认与应收账款回收。",
    ]
    metas = [
        {"stock_code": "000001", "year": 2023, "source": "mda"},
        {"stock_code": "000001", "year": 2023, "source": "mda"},
        {"stock_code": "000002", "year": 2023, "source": "mda"},
        {"stock_code": "000002", "year": 2022, "source": "audit"},
    ]
    ids = ["doc_a", "doc_b", "doc_c", "doc_d"]
    return texts, metas, ids


def test_ch13_embed_and_store_retrieve_end_to_end():
    base = Path(__file__).resolve().parent / f"_tmp_ch13_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=False)
    try:
        texts, metas, ids = _sample_docs()
        model = load_model(dim=128)
        vecs, log = embed_texts(texts, model=model, return_log=True)
        assert vecs.shape == (4, 128)
        assert 0.0 <= log["empty_ratio"] <= 1.0

        store = create_store(persist_dir=base)
        add_documents(store, vecs, metas, texts, ids=ids)
        stats = get_stats(store)
        assert stats["n_docs"] == 4
        assert stats["dim"] == 128

        res = retrieve("应收账款坏账风险", model, store, top_k=2, threshold=-1.0)
        assert len(res) == 2
        assert any("应收账款" in r["document"] for r in res)

        filtered = retrieve_with_filter("审计事项", model, store, top_k=3, threshold=-1.0, source="audit")
        assert len(filtered) == 1
        assert filtered[0]["metadata"]["source"] == "audit"

        save(store)
        loaded = load(base)
        res2 = retrieve("供应链风险", model, loaded, top_k=2, threshold=-1.0)
        assert len(res2) >= 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_ch13_recall_and_rerank():
    texts, metas, ids = _sample_docs()
    model = load_model(dim=96)
    vecs = embed_texts(texts, model=model)
    store = create_store()
    add_documents(store, vecs, metas, texts, ids=ids)

    queries = ["供应链风险", "研发费用率提升"]
    relevant = [{"doc_a"}, {"doc_c"}]
    scores = eval_recall(queries, relevant, model, store, k_list=[1, 2, 3])
    assert set(scores.keys()) == {"recall@1", "recall@2", "recall@3"}
    assert 0.0 <= scores["recall@1"] <= 1.0

    raw = retrieve("应收账款", model, store, top_k=4, threshold=-1.0)
    rr = rerank(raw, max_per_stock=1)
    counts = {}
    for r in rr:
        code = r["metadata"]["stock_code"]
        counts[code] = counts.get(code, 0) + 1
    assert all(v <= 1 for v in counts.values())


def test_ch13_invalid_inputs_raise():
    model = load_model(dim=64)
    try:
        embed_texts(["x"], model=model, batch_size=0)
        raised_batch = False
    except ValueError:
        raised_batch = True
    assert raised_batch

    store = create_store()
    try:
        add_documents(
            store,
            np.zeros((1, 64), dtype=np.float32),
            [{"stock_code": "000001"}],
            ["abc"],
        )
        raised_meta = False
    except ValueError:
        raised_meta = True
    assert raised_meta
