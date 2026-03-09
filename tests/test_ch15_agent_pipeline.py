from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from ifa.agent import (
    build_evidence_table,
    build_report,
    create_agent,
    export_markdown,
    export_pdf,
    orchestrate,
    run_agent,
)


def test_ch15_single_agent_run_schema():
    tools = {
        "get_indicators": lambda code, start_year, end_year: {
            "code": code,
            "years": [start_year, end_year],
            "current_ratio": 1.8,
        },
        "get_risk_score": lambda stock_code: {"stock_code": stock_code, "risk_score": 0.23},
        "get_esg_score": lambda stock_code: {"stock_code": stock_code, "total_score": 0.71},
    }

    agent = create_agent("analyst", "financial_analysis", tools=tools)
    result = run_agent(
        agent,
        task="analyze stock",
        context={"stock_code": "000001", "start_year": 2021, "end_year": 2023},
    )

    assert result["status"] == "success"
    assert {"summary", "evidence", "warnings", "tool_traces", "status"}.issubset(result.keys())
    assert len(result["tool_traces"]) == 3
    assert result["warnings"] == []


def test_ch15_multi_agent_retry_and_status():
    state = {"calls": 0}

    def flaky_indicators(code, start_year, end_year):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("temporary failure")
        return {"code": code, "panel_ok": True, "years": [start_year, end_year]}

    tools = {
        "get_indicators": flaky_indicators,
        "get_risk_score": lambda stock_code: {"risk_score": 0.31, "stock_code": stock_code},
        "get_esg_score": lambda stock_code: {"esg_score": 0.66, "stock_code": stock_code},
    }

    out = orchestrate("000001", tools=tools, max_retry=2)
    assert out["status"] == "success"
    assert out["review_result"]["passed"] is True
    assert out["retries"] == 1
    assert len(out["orchestration_log"]) >= 2


def test_ch15_report_build_and_exports():
    fake_results = {
        "analysis_result": {
            "summary": "analysis done",
            "evidence": {
                "get_indicators": {"current_ratio": 1.7},
                "get_risk_score": {"risk_score": 0.28},
            },
            "warnings": [],
            "tool_traces": [{"tool": "get_indicators", "status": "ok"}],
            "status": "success",
        },
        "review_result": {"passed": True, "summary": "review passed", "issues": []},
        "status": "success",
    }

    evidence_df = build_evidence_table(fake_results)
    assert len(evidence_df) == 2

    report_md = build_report(fake_results, meta={"stock_code": "000001", "chapter": "ch15"})
    assert "# Automated Financial Analysis Report" in report_md
    assert "## Evidence Map" in report_md

    base = Path(__file__).resolve().parent / f"_tmp_ch15_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=False)
    try:
        md_path = export_markdown(report_md, base / "report.md")
        pdf_path = export_pdf(report_md, base / "report.pdf")

        assert md_path.exists()
        assert pdf_path.exists()
        assert md_path.read_text(encoding="utf-8").startswith("# Automated Financial Analysis Report")
    finally:
        shutil.rmtree(base, ignore_errors=True)
