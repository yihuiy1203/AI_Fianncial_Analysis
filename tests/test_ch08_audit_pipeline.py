from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from ifa.audit import extract_kam, extract_metadata, extract_opinion_type, parse_pdf, summarize_kam


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        self.calls += 1
        if self.calls == 1:
            return "not-a-json"
        return json.dumps(
            {
                "topic": "商誉减值",
                "risk_point": "并购标的业绩不及预期导致减值风险上升。",
                "audit_response": "执行减值测试并复核关键参数。",
                "one_line_summary": "公司商誉减值测试存在高判断风险。",
            },
            ensure_ascii=False,
        )


def _write_report(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_ch08_pipeline_normal_case():
    base = Path(__file__).resolve().parent / f"_tmp_ch08_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=False)
    try:
        report = base / "000001_2023_audit_report.txt"
        _write_report(
            report,
            """
            审计报告

            我们发表保留意见。
            由于存货盘点受限，我们无法获取充分证据。

            关键审计事项

            事项一 收入确认
            公司收入确认涉及多履约义务拆分，存在跨期确认风险。

            事项二 商誉减值
            管理层对未来现金流预测存在主观判断，减值测试不确定性较高。

            其他信息
            管理层负责其他信息披露。
            """,
        )

        parsed = parse_pdf(report)
        paragraphs = parsed["paragraphs"]
        assert len(paragraphs) > 0

        opinion = extract_opinion_type(paragraphs)
        assert opinion["opinion_type"] == "保留意见"
        assert opinion["evidence_page"] == 1

        kam_items = extract_kam(paragraphs)
        assert len(kam_items) >= 2
        assert "收入确认" in kam_items[0]["kam_text"]

        meta = extract_metadata(paragraphs, file_name=report.name)
        assert meta["stock_code"] == "000001"
        assert meta["year"] == 2023

        s = summarize_kam(kam_items[0]["kam_text"], llm_client=_FakeLLM())
        assert {"topic", "risk_point", "audit_response", "one_line_summary", "raw_response"}.issubset(s.keys())
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_ch08_pipeline_boundary_no_kam():
    paragraphs = [
        {"page_no": 1, "source_id": 1, "text": "我们发表标准无保留意见。"},
        {"page_no": 1, "source_id": 2, "text": "管理层责任"},
    ]
    opinion = extract_opinion_type(paragraphs)
    kam_items = extract_kam(paragraphs)
    summary = summarize_kam("", llm_client=None)

    assert opinion["opinion_type"] == "标准无保留意见"
    assert kam_items == []
    assert summary["one_line_summary"] == "无可用KAM文本"


def test_ch08_pipeline_failure_missing_file():
    missing = Path(__file__).resolve().parent / "_not_exists_ch08_report.pdf"
    try:
        parse_pdf(missing)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
