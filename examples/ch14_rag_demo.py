from __future__ import annotations

import json

from ifa.rag import (
    add_metadata,
    chunk_by_semantic,
    deduplicate_chunks,
    eval_batch,
    generate_eval_report,
    run_rag,
)


class DemoRetriever:
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
        # simple ranking by keyword overlap count
        scored = []
        for d in out:
            score = sum(1 for w in ["供应链", "研发", "风险", "审计"] if w in query and w in d["document"])
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]


def demo_llm(prompt: str):
    if "无可用上下文" in prompt:
        return {"text": "未找到足够依据。", "citations": []}
    # return deterministic answer with citation
    return {
        "text": "结论：文本显示公司提及供应链风险[1]。\n关键证据：[1]\n不确定性：仅基于当前检索片段。",
        "citations": [1],
    }


def main() -> None:
    raw_text = (
        "公司在2023年年报管理层讨论中提示上游原材料价格波动，可能带来供应链风险。\n\n"
        "公司表示研发投入持续提升，重点投向核心产品迭代。\n\n"
        "审计报告强调收入确认与应收回款可回收性。"
    )

    chunks = chunk_by_semantic(raw_text)
    chunks_meta = add_metadata(chunks, source="mda", year=2023, section="MD&A", base_id="demo")
    chunks_meta = deduplicate_chunks(chunks_meta)

    docs = [
        {
            "id": c["chunk_id"],
            "document": c["text"],
            "metadata": {
                "source": c["source"],
                "year": c["year"],
                "section": c["section"],
                "stock_code": "000001",
            },
        }
        for c in chunks_meta
    ]

    retriever = DemoRetriever(docs)
    q1 = run_rag("公司是否提及供应链风险？", retriever, demo_llm, top_k=3)
    q2 = run_rag("审计关注了什么事项？", retriever, demo_llm, top_k=3)

    batch_df = eval_batch([q1, q2])
    report = generate_eval_report([q1, q2])

    print(
        json.dumps(
            {
                "n_chunks": len(docs),
                "q1_answer": q1["answer"],
                "q1_citations": q1["citation_ids"],
                "q2_answer": q2["answer"],
                "batch_rows": int(len(batch_df)),
                "report": {
                    "n": report["n"],
                    "faithfulness_mean": report["faithfulness_mean"],
                    "relevance_mean": report["relevance_mean"],
                    "citation_rate": report["citation_rate"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
