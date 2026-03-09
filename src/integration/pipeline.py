from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

Tool = Callable[..., Any]

DEFAULT_LAYER_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5"]


@dataclass(frozen=True)
class StepResult:
    name: str
    status: str
    latency_ms: int
    detail: str


def check_layer_dependencies(
    layer_graph: dict[str, list[str]],
    allowed_order: list[str] | None = None,
) -> dict[str, Any]:
    order = allowed_order or DEFAULT_LAYER_ORDER
    position = {name: idx for idx, name in enumerate(order)}
    violations: list[dict[str, str]] = []

    for layer, deps in layer_graph.items():
        if layer not in position:
            violations.append({"layer": layer, "dependency": "*", "reason": "unknown_layer"})
            continue
        for dep in deps:
            if dep not in position:
                violations.append({"layer": layer, "dependency": dep, "reason": "unknown_dependency"})
                continue
            if position[dep] > position[layer]:
                violations.append({"layer": layer, "dependency": dep, "reason": "upward_dependency"})

    return {"passed": len(violations) == 0, "violations": violations}


def _run_step(name: str, tool: Tool, payload: dict[str, Any]) -> StepResult:
    t0 = perf_counter()
    try:
        out = tool(**payload)
        detail = f"ok:{type(out).__name__}"
        status = "ok"
    except Exception as exc:  # pragma: no cover - defensive path
        detail = f"error:{exc}"
        status = "error"
    return StepResult(name=name, status=status, latency_ms=int((perf_counter() - t0) * 1000), detail=detail)


def run_integration_smoke(
    stock_code: str,
    tools: dict[str, Tool],
    strict: bool = False,
) -> dict[str, Any]:
    required = ["load_data", "build_indicators", "run_analysis", "build_report"]
    steps: list[dict[str, Any]] = []
    ctx: dict[str, Any] = {"stock_code": stock_code}

    for name in required:
        if name not in tools:
            step = StepResult(name=name, status="missing", latency_ms=0, detail="tool_not_registered")
            steps.append(step.__dict__)
            if strict:
                break
            continue

        payload = {"stock_code": stock_code, "context": dict(ctx)}
        result = _run_step(name, tools[name], payload)
        steps.append(result.__dict__)
        if result.status != "ok" and strict:
            break
        ctx[name] = result.detail

    n_ok = sum(1 for s in steps if s["status"] == "ok")
    status = "success" if n_ok == len(required) else ("partial_success" if n_ok > 0 else "failed")
    if any(s["status"] == "error" for s in steps):
        status = "failed"

    return {
        "stock_code": stock_code,
        "steps": steps,
        "n_ok": n_ok,
        "n_total": len(required),
        "status": status,
    }


def summarize_test_results(test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(test_rows)
    passed = sum(1 for row in test_rows if bool(row.get("passed")))
    failed = total - passed
    pass_rate = 0.0 if total == 0 else passed / total
    failed_cases = [str(row.get("name", "")) for row in test_rows if not bool(row.get("passed"))]
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "failed_cases": failed_cases,
    }
