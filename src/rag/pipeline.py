from __future__ import annotations

import re
import time
from typing import Any, Callable


SYSTEM_TEMPLATE = (
    "你是财务分析助手。请只基于给定上下文回答。\n"
    "输出结构：\n"
    "1) 结论\n"
    "2) 关键证据（引用[编号]）\n"
    "3) 不确定性与边界\n"
    "若证据不足，请明确写“未找到足够依据”。"
)


def _extract_citations(text: str) -> list[int]:
    hits = re.findall(r"\[(\d+)\]", text)
    return sorted({int(x) for x in hits})


def build_prompt(query: str, docs: list[dict[str, Any]], system_template: str | None = None) -> str:
    sys = system_template or SYSTEM_TEMPLATE
    lines = []
    for i, d in enumerate(docs, start=1):
        lines.append(f"[{i}] {d.get('document', d.get('text', ''))}")
    context = "\n".join(lines) if lines else "[无可用上下文]"
    return f"{sys}\n\n上下文：\n{context}\n\n用户问题：{query}"


def call_llm(prompt: str, llm_caller: Callable[[str], Any]) -> dict[str, Any]:
    raw = llm_caller(prompt)
    if isinstance(raw, dict):
        text = str(raw.get("text", "")).strip()
        citations = raw.get("citations", [])
        if not isinstance(citations, list):
            citations = []
        return {"text": text, "citations": citations}

    text = str(raw).strip()
    return {"text": text, "citations": _extract_citations(text)}


def format_answer(answer_text: str, docs: list[dict[str, Any]], citations: list[int] | None = None) -> dict[str, Any]:
    citations = citations or _extract_citations(answer_text)
    citation_objs = []
    for c in citations:
        if 1 <= c <= len(docs):
            d = docs[c - 1]
            citation_objs.append(
                {
                    "index": c,
                    "chunk_id": d.get("id", d.get("chunk_id", "")),
                    "source": d.get("metadata", {}).get("source", d.get("source", "")),
                    "stock_code": d.get("metadata", {}).get("stock_code"),
                    "year": d.get("metadata", {}).get("year"),
                }
            )
    return {
        "answer": answer_text,
        "citations": citation_objs,
        "citation_ids": citations,
        "retrieved_chunks": docs,
    }


def _run_retrieve(query: str, retriever: Any, top_k: int, filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query, top_k=top_k, filters=filters)
    if callable(retriever):
        return retriever(query=query, top_k=top_k, filters=filters)
    raise ValueError("retriever must be callable or object with .retrieve")


def run_rag(
    query: str,
    retriever: Any,
    llm_caller: Callable[[str], Any],
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    system_template: str | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    t0 = time.time()
    docs = _run_retrieve(query, retriever, top_k=top_k, filters=filters)
    prompt = build_prompt(query, docs, system_template=system_template)
    llm_out = call_llm(prompt, llm_caller)
    result = format_answer(llm_out["text"], docs, citations=llm_out.get("citations"))

    warnings = []
    if not docs:
        warnings.append("empty_retrieval")
    if not result["citations"]:
        warnings.append("missing_citations")

    result.update(
        {
            "query": query,
            "latency_ms": int((time.time() - t0) * 1000),
            "warnings": warnings,
            "prompt": prompt,
        }
    )
    return result
