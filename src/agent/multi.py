from __future__ import annotations

from typing import Any, Callable

from .report import build_report
from .single import Agent, create_agent, run_agent

Tool = Callable[..., Any]


def create_analyst_agent(tools: dict[str, Tool] | None = None) -> Agent:
    return create_agent(
        name="analyst",
        role="financial_analysis",
        tools=tools,
        constraints=["provide evidence", "explicit uncertainty"],
    )


def create_reviewer_agent(tools: dict[str, Tool] | None = None) -> Agent:
    return create_agent(
        name="reviewer",
        role="quality_review",
        tools=tools,
        constraints=["verify consistency", "check missing evidence"],
    )


def create_writer_agent(tools: dict[str, Tool] | None = None) -> Agent:
    return create_agent(
        name="writer",
        role="report_generation",
        tools=tools,
        constraints=["clear structure", "traceable evidence"],
    )


def _review_result(analysis_result: dict[str, Any], tools: dict[str, Tool]) -> dict[str, Any]:
    if "review_analysis" in tools:
        out = tools["review_analysis"](analysis_result=analysis_result)
        passed = bool(out.get("passed", False))
        return {
            "passed": passed,
            "summary": str(out.get("summary", "")),
            "issues": list(out.get("issues", [])),
        }

    statuses = [t.get("status") for t in analysis_result.get("tool_traces", [])]
    has_error = any(s in {"error", "missing"} for s in statuses)
    has_evidence = bool(analysis_result.get("evidence"))
    passed = has_evidence and not has_error
    issues = []
    if not has_evidence:
        issues.append("no_evidence")
    if has_error:
        issues.append("tool_failures")
    return {
        "passed": passed,
        "summary": "review passed" if passed else "review failed",
        "issues": issues,
    }


def orchestrate(
    stock_code: str,
    tools: dict[str, Tool] | dict[str, dict[str, Tool]],
    max_retry: int = 2,
    start_year: int = 2021,
    end_year: int = 2023,
) -> dict[str, Any]:
    if max_retry < 0:
        raise ValueError("max_retry must be >= 0")

    if isinstance(tools.get("analyst"), dict):  # type: ignore[union-attr]
        analyst_tools = dict(tools.get("analyst", {}))  # type: ignore[union-attr]
        reviewer_tools = dict(tools.get("reviewer", {}))  # type: ignore[union-attr]
        writer_tools = dict(tools.get("writer", {}))  # type: ignore[union-attr]
    else:
        analyst_tools = dict(tools)  # type: ignore[arg-type]
        reviewer_tools = dict(tools)  # type: ignore[arg-type]
        writer_tools = dict(tools)  # type: ignore[arg-type]

    analyst = create_analyst_agent(analyst_tools)
    reviewer = create_reviewer_agent(reviewer_tools)
    writer = create_writer_agent(writer_tools)

    log: list[dict[str, Any]] = []
    retries = 0
    analysis_result: dict[str, Any] = {}
    review_result: dict[str, Any] = {"passed": False, "issues": ["not_run"], "summary": "not_run"}

    while retries <= max_retry:
        analysis_result = run_agent(
            analyst,
            task=f"analyze stock {stock_code}",
            context={"stock_code": stock_code, "start_year": start_year, "end_year": end_year},
        )
        review_result = _review_result(analysis_result, reviewer.tools)
        log.append({"round": retries + 1, "review_passed": review_result["passed"], "issues": review_result["issues"]})
        if review_result["passed"]:
            break
        retries += 1

    report_payload = {
        "analysis_result": analysis_result,
        "review_result": review_result,
        "orchestration_log": log,
    }

    if "compose_report" in writer.tools:
        final_report = writer.tools["compose_report"](report_payload=report_payload, stock_code=stock_code)
    else:
        report_md = build_report(report_payload, meta={"stock_code": stock_code})
        final_report = {"markdown": report_md}

    status = "success" if review_result.get("passed") else "needs_manual_review"

    return {
        "analysis_result": analysis_result,
        "review_result": review_result,
        "final_report": final_report,
        "orchestration_log": log,
        "retries": retries,
        "status": status,
    }
