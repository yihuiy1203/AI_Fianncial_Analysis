from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from ifa.integration import (
    build_release_checklist,
    bump_version,
    check_layer_dependencies,
    export_markdown,
    export_release_manifest,
    generate_api_markdown,
    generate_quickstart,
    parse_semver,
    run_integration_smoke,
    summarize_test_results,
    validate_release_ready,
)


def test_ch16_layer_dependency_rules():
    ok_graph = {"L0": [], "L1": ["L0"], "L2": ["L0", "L1"], "L3": ["L0", "L2"], "L4": ["L0", "L2"], "L5": ["L0", "L2", "L3", "L4"]}
    bad_graph = {"L2": ["L3"]}  # upward dependency

    ok = check_layer_dependencies(ok_graph)
    bad = check_layer_dependencies(bad_graph)

    assert ok["passed"] is True
    assert bad["passed"] is False
    assert bad["violations"][0]["reason"] == "upward_dependency"


def test_ch16_integration_smoke_and_test_summary():
    tools = {
        "load_data": lambda stock_code, context: {"stock_code": stock_code, "rows": 12},
        "build_indicators": lambda stock_code, context: {"code": stock_code, "n_metrics": 8},
        "run_analysis": lambda stock_code, context: {"risk": 0.22, "esg": 0.71},
        "build_report": lambda stock_code, context: {"status": "ok", "report_id": f"r-{stock_code}"},
    }
    smoke = run_integration_smoke("000001", tools=tools)
    assert smoke["status"] == "success"
    assert smoke["n_ok"] == 4

    summary = summarize_test_results(
        [
            {"name": "unit", "passed": True},
            {"name": "integration", "passed": True},
            {"name": "regression", "passed": False},
        ]
    )
    assert summary["total"] == 3
    assert summary["failed"] == 1
    assert "regression" in summary["failed_cases"]


def test_ch16_docs_and_release_workflow():
    base = Path(__file__).resolve().parent / f"_tmp_ch16_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=False)
    try:
        api_md = generate_api_markdown(["ifa.integration.pipeline", "ifa.integration.release"], title="Ch16 API")
        quickstart = generate_quickstart()
        assert "# Ch16 API" in api_md
        assert "pytest -q" in quickstart

        api_path = export_markdown(api_md, base / "api.md")
        qs_path = export_markdown(quickstart, base / "quickstart.md")
        assert api_path.exists()
        assert qs_path.exists()

        major, minor, patch = parse_semver("v3.2.0")
        assert (major, minor, patch) == (3, 2, 0)
        assert bump_version("v3.2.0", "minor") == "v3.3.0"

        checklist = build_release_checklist(
            version="v4.0.0",
            test_summary={"failed": 0},
            doc_paths=[api_path, qs_path],
            artifacts=[api_path],
        )
        gate = validate_release_ready(checklist)
        assert gate["ready"] is True

        manifest = export_release_manifest(
            {"version": "v4.0.0", "checklist": checklist, "ready": gate["ready"]},
            base / "release_manifest.json",
        )
        assert manifest.exists()
    finally:
        shutil.rmtree(base, ignore_errors=True)
