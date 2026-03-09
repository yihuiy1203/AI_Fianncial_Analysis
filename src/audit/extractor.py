from __future__ import annotations

import re
from typing import Iterable


OPINION_RULES: list[tuple[str, str]] = [
    ("无法表示意见", "无法表示意见"),
    ("否定意见", "否定意见"),
    ("无保留意见", "标准无保留意见"),
    ("标准无保留", "标准无保留意见"),
    ("保留意见", "保留意见"),
]

SECTION_STOP_WORDS = [
    "其他信息",
    "管理层责任",
    "注册会计师责任",
    "强调事项",
    "其他事项",
]


def _ensure_paragraphs(paragraphs: Iterable[dict]) -> list[dict]:
    out = list(paragraphs)
    if not out:
        return out
    for p in out:
        if "text" not in p:
            raise ValueError("paragraph item missing 'text'")
    return out


def extract_opinion_type(paragraphs: Iterable[dict]) -> dict[str, object]:
    items = _ensure_paragraphs(paragraphs)
    for p in items:
        text = str(p.get("text", ""))
        for key, label in OPINION_RULES:
            if key in text:
                return {
                    "opinion_type": label,
                    "evidence_text": text[:160],
                    "evidence_page": int(p.get("page_no", -1)),
                    "rule_hit": key,
                }
    return {
        "opinion_type": "标准无保留意见",
        "evidence_text": "",
        "evidence_page": -1,
        "rule_hit": "default",
    }


def extract_kam(paragraphs: Iterable[dict]) -> list[dict[str, object]]:
    items = _ensure_paragraphs(paragraphs)
    if not items:
        return []

    start_idx = -1
    for i, p in enumerate(items):
        if "关键审计事项" in str(p.get("text", "")):
            start_idx = i
            break
    if start_idx < 0:
        return []

    content: list[dict] = []
    for p in items[start_idx + 1 :]:
        t = str(p.get("text", "")).strip()
        if any(stop in t for stop in SECTION_STOP_WORDS):
            break
        if t:
            content.append(p)

    if not content:
        return []

    # Split KAM items by headings like "事项一" or numbered lines.
    groups: list[list[dict]] = []
    current: list[dict] = []
    split_pattern = re.compile(r"^(事项[一二三四五六七八九十]+|[0-9]+[、.])")
    for p in content:
        t = str(p["text"]).strip()
        if split_pattern.match(t) and current:
            groups.append(current)
            current = [p]
        else:
            current.append(p)
    if current:
        groups.append(current)

    result: list[dict[str, object]] = []
    for idx, group in enumerate(groups, start=1):
        text = "\n".join(str(x["text"]).strip() for x in group if str(x["text"]).strip())
        page = int(group[0].get("page_no", -1))
        first_line = text.splitlines()[0] if text else f"KAM-{idx}"
        result.append(
            {
                "kam_id": idx,
                "kam_title": first_line[:80],
                "kam_text": text,
                "evidence_page": page,
            }
        )
    return result


def extract_metadata(paragraphs: Iterable[dict], file_name: str = "") -> dict[str, object]:
    items = _ensure_paragraphs(paragraphs)
    joined = "\n".join(str(p.get("text", "")) for p in items)

    stock = ""
    year = None
    m_stock = re.search(r"(\d{6})", file_name) or re.search(r"(\d{6})", joined)
    if m_stock:
        stock = m_stock.group(1)

    m_year = re.search(r"(20\d{2})", file_name) or re.search(r"(20\d{2})年", joined)
    if m_year:
        year = int(m_year.group(1))

    auditor = ""
    m_auditor = re.search(r"([\u4e00-\u9fa5A-Za-z（）()]{2,40}会计师事务所)", joined)
    if m_auditor:
        auditor = m_auditor.group(1)

    return {"stock_code": stock, "year": year, "auditor": auditor}
