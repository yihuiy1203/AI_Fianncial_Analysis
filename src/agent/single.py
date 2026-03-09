from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

Tool = Callable[..., Any]


@dataclass
class Agent:
    name: str
    role: str
    tools: dict[str, Tool] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)


def _default_tool_registry() -> dict[str, Tool]:
    # Keep defaults lazy and lightweight; callers can fully override in tests/examples.
    return {}


def register_tools(extra_tools: dict[str, Tool] | None = None) -> dict[str, Tool]:
    tools = _default_tool_registry()
    if extra_tools:
        for name, tool in extra_tools.items():
            if not callable(tool):
                raise ValueError(f"tool '{name}' is not callable")
            tools[name] = tool
    return tools


def create_agent(
    name: str,
    role: str,
    tools: dict[str, Tool] | None = None,
    constraints: list[str] | None = None,
) -> Agent:
    return Agent(
        name=name,
        role=role,
        tools=register_tools(tools),
        constraints=list(constraints or []),
    )


def _build_plan(agent: Agent, task: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = context.get("plan")
    if isinstance(explicit, list):
        return explicit

    stock_code = context.get("stock_code")
    start_year = int(context.get("start_year", 2021))
    end_year = int(context.get("end_year", 2023))

    plan = []
    if "get_indicators" in agent.tools:
        plan.append(
            {
                "tool": "get_indicators",
                "kwargs": {"code": stock_code, "start_year": start_year, "end_year": end_year},
            }
        )
    if "get_risk_score" in agent.tools:
        plan.append({"tool": "get_risk_score", "kwargs": {"stock_code": stock_code}})
    if "get_esg_score" in agent.tools:
        plan.append({"tool": "get_esg_score", "kwargs": {"stock_code": stock_code}})
    if "query_knowledge" in agent.tools:
        plan.append({"tool": "query_knowledge", "kwargs": {"query": task, "stock_code": stock_code}})

    if not plan:
        for name in agent.tools:
            plan.append({"tool": name, "kwargs": {"task": task, "context": context}})
    return plan


def _invoke_tool(tool: Tool, kwargs: dict[str, Any], context: dict[str, Any]) -> Any:
    try:
        return tool(**kwargs)
    except TypeError:
        merged = dict(context)
        merged.update(kwargs)
        return tool(**merged)


def finalize_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": str(payload.get("summary", "")),
        "evidence": payload.get("evidence", {}),
        "warnings": list(payload.get("warnings", [])),
        "tool_traces": list(payload.get("tool_traces", [])),
        "status": str(payload.get("status", "unknown")),
        "task": payload.get("task", ""),
        "agent": payload.get("agent", ""),
    }


def run_agent(agent: Agent, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    ctx = dict(context or {})
    plan = _build_plan(agent, task, ctx)

    traces: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    warnings: list[str] = []

    for step in plan:
        tool_name = str(step.get("tool", ""))
        kwargs = dict(step.get("kwargs", {}))
        t0 = perf_counter()

        if tool_name not in agent.tools:
            traces.append(
                {
                    "tool": tool_name,
                    "status": "missing",
                    "latency_ms": int((perf_counter() - t0) * 1000),
                    "args": kwargs,
                    "error": "tool_not_registered",
                }
            )
            warnings.append(f"missing_tool:{tool_name}")
            continue

        tool = agent.tools[tool_name]
        try:
            output = _invoke_tool(tool, kwargs, ctx)
            evidence[tool_name] = output
            traces.append(
                {
                    "tool": tool_name,
                    "status": "ok",
                    "latency_ms": int((perf_counter() - t0) * 1000),
                    "args": kwargs,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive path
            traces.append(
                {
                    "tool": tool_name,
                    "status": "error",
                    "latency_ms": int((perf_counter() - t0) * 1000),
                    "args": kwargs,
                    "error": str(exc),
                }
            )
            warnings.append(f"tool_error:{tool_name}")

    n_ok = sum(1 for t in traces if t["status"] == "ok")
    n_total = len(traces)
    if n_total == 0:
        status = "failed"
    elif n_ok == n_total:
        status = "success"
    elif n_ok > 0:
        status = "partial_success"
    else:
        status = "failed"

    summary = (
        f"{agent.name} completed task with {n_ok}/{n_total} successful tool calls."
        if n_total
        else f"{agent.name} had no executable plan."
    )

    return finalize_result(
        {
            "agent": agent.name,
            "task": task,
            "summary": summary,
            "evidence": evidence,
            "warnings": warnings,
            "tool_traces": traces,
            "status": status,
        }
    )
