from __future__ import annotations

import pandas as pd

from ifa.rag import (
    add_metadata,
    chunk_by_length,
    chunk_by_semantic,
    deduplicate_chunks,
    eval_batch,
    eval_faithfulness,
    eval_relevance,
    generate_eval_report,
    run_rag,
)


class _FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query: str, top_k: int = 5, filters=None):
        out = self.docs
        if filters:
            tmp = []
            for d in out:
                ok = True
                for k, v in filters.items():
                    if d.get("metadata", {}).get(k) != v:
                        ok = False
                        break
                if ok:
                    tmp.append(d)
            out = tmp
        return out[:top_k]


def _fake_llm(prompt: str):
    if "无可用上下文" in prompt:
        return {"text": "未找到足够依据。", "citations": []}
    return {"text": "结论：公司提及供应链风险[1]。不确定性：仅基于当前文本。", "citations": [1]}


def test_ch14_chunker_and_metadata_pipeline():
    text = "第一段：公司提示供应链风险。\n\n第二段：研发投入持续增加。\n\n第二段：研发投入持续增加。"
    chunks = chunk_by_length(text, chunk_size=14, overlap=4)
    assert len(chunks) >= 2

    sem = chunk_by_semantic(text)
    meta = add_metadata(sem, source="mda", year=2023, section="MD&A", base_id="mda")
    assert all("chunk_id" in m and "start_offset" in m for m in meta)

    dedup = deduplicate_chunks(meta)
    assert len(dedup) < len(meta)


def test_ch14_run_rag_with_citations_and_filter():
    docs = [
        {
            "id": "c1",
            "document": "公司在2023年年报中提及供应链风险和原材料波动。",
            "metadata": {"source": "mda", "stock_code": "000001", "year": 2023},
        },
        {
            "id": "c2",
            "document": "审计报告关注收入确认事项。",
            "metadata": {"source": "audit", "stock_code": "000001", "year": 2022},
        },
    ]
    retriever = _FakeRetriever(docs)

    res = run_rag("公司是否提及供应链风险？", retriever, _fake_llm, top_k=2, filters={"source": "mda"})
    assert "answer" in res and res["answer"]
    assert len(res["retrieved_chunks"]) == 1
    assert len(res["citations"]) == 1
    assert res["warnings"] == []


def test_ch14_eval_functions_and_report():
    contexts = ["公司提及供应链风险并解释了上游波动影响。"]
    ans = "公司提及供应链风险，存在不确定性。"
    f = eval_faithfulness(ans, contexts)
    r = eval_relevance("供应链风险？", ans)
    assert 0.0 <= f <= 1.0
    assert 0.0 <= r <= 1.0

    records = [
        {
            "query": "供应链风险？",
            "answer": ans,
            "retrieved_chunks": [{"document": contexts[0]}],
            "citations": [{"index": 1}],
        },
        {
            "query": "研发趋势？",
            "answer": "未找到足够依据。",
            "retrieved_chunks": [],
            "citations": [],
        },
    ]

    df = eval_batch(records)
    assert isinstance(df, pd.DataFrame)
    assert {"faithfulness", "relevance", "n_context", "n_citations"}.issubset(df.columns)

    report = generate_eval_report(records)
    assert report["n"] == 2
    assert 0.0 <= report["citation_rate"] <= 1.0
