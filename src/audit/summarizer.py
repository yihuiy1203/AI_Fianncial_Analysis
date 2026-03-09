from __future__ import annotations

import json
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str, temperature: float = 0.2) -> str: ...


REQUIRED_FIELDS = ["topic", "risk_point", "audit_response", "one_line_summary"]


def _default_summary(kam_text: str) -> dict[str, str]:
    short = kam_text.strip().replace("\n", " ")
    short = short[:120] if short else ""
    return {
        "topic": "未分类事项",
        "risk_point": short,
        "audit_response": "建议结合原文核查关键会计判断与证据。",
        "one_line_summary": short or "无可用KAM文本",
    }


def _safe_json_load(text: str) -> dict:
    try:
        obj = json.loads(text)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def validate_summary(data: dict, kam_text: str) -> dict[str, str]:
    base = _default_summary(kam_text)
    out = {}
    for f in REQUIRED_FIELDS:
        val = str(data.get(f, "")).strip()
        out[f] = val if val else base[f]

    # Keep summary concise and deterministic.
    out["one_line_summary"] = out["one_line_summary"][:160]
    return out


def summarize_kam(kam_text: str, llm_client: LLMClient | None = None) -> dict[str, str]:
    if not kam_text or not kam_text.strip():
        return validate_summary({}, "")

    if llm_client is None:
        return validate_summary({}, kam_text)

    prompt = (
        "仅依据给定文本，输出JSON，字段必须为: "
        "topic, risk_point, audit_response, one_line_summary。"
        "不得补充原文未出现的关键信息。\n"
        f"文本:\n{kam_text}"
    )

    first = llm_client.generate(prompt=prompt, temperature=0.2)
    data = _safe_json_load(first)
    if not data or any(k not in data for k in REQUIRED_FIELDS):
        second = llm_client.generate(prompt=prompt, temperature=0.0)
        data = _safe_json_load(second)

    out = validate_summary(data, kam_text)
    out["raw_response"] = first
    return out
