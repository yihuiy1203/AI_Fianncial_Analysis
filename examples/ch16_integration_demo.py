from __future__ import annotations

import json
from pathlib import Path

from ifa.integration import (
    build_release_checklist,
    export_markdown,
    export_release_manifest,
    generate_api_markdown,
    generate_quickstart,
    run_integration_smoke,
    summarize_test_results,
    validate_release_ready,
)


def main() -> None:
    smoke_tools = {
        "load_data": lambda stock_code, context: {"stock_code": stock_code, "rows": 15},
        "build_indicators": lambda stock_code, context: {"metric_count": 10, "stock_code": stock_code},
        "run_analysis": lambda stock_code, context: {"risk_score": 0.24, "esg_score": 0.68},
        "build_report": lambda stock_code, context: {"report_id": f"ifa-{stock_code}-v4"},
    }

    smoke = run_integration_smoke("000001", smoke_tools)
    test_summary = summarize_test_results(
        [
            {"name": "ch16_smoke", "passed": smoke["status"] == "success"},
            {"name": "api_doc_generation", "passed": True},
        ]
    )

    api_md = generate_api_markdown(["ifa.integration.pipeline", "ifa.integration.docs", "ifa.integration.release"], title="IFA Ch16 API")
    quickstart_md = generate_quickstart()

    out_dir = Path("outputs/reports")
    api_path = export_markdown(api_md, out_dir / "ch16_api_reference.md")
    quickstart_path = export_markdown(quickstart_md, out_dir / "ch16_quickstart.md")

    checklist = build_release_checklist(
        version="v4.0.0",
        test_summary=test_summary,
        doc_paths=[api_path, quickstart_path],
        artifacts=[api_path],
    )
    release_gate = validate_release_ready(checklist)
    manifest_path = export_release_manifest(
        {"version": "v4.0.0", "smoke": smoke, "test_summary": test_summary, "checklist": checklist, "release_gate": release_gate},
        out_dir / "ch16_release_manifest.json",
    )

    print(
        json.dumps(
            {
                "smoke_status": smoke["status"],
                "tests_failed": test_summary["failed"],
                "release_ready": release_gate["ready"],
                "api_doc": str(api_path),
                "quickstart": str(quickstart_path),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
