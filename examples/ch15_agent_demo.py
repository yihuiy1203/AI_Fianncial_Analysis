from __future__ import annotations

import json
from pathlib import Path

from ifa.agent import build_report, export_markdown, orchestrate


def main() -> None:
    tools = {
        "get_indicators": lambda code, start_year, end_year: {
            "stock_code": code,
            "window": [start_year, end_year],
            "current_ratio": 1.92,
            "debt_ratio": 0.47,
        },
        "get_risk_score": lambda stock_code: {
            "stock_code": stock_code,
            "risk_score": 0.22,
            "level": "low",
        },
        "get_esg_score": lambda stock_code: {
            "stock_code": stock_code,
            "total_score": 0.74,
            "grade": "A-",
        },
    }

    out = orchestrate("000001", tools=tools, max_retry=1)
    report_md = build_report(out, meta={"stock_code": "000001", "chapter": "ch15", "pipeline": "multi-agent"})

    out_dir = Path("outputs/reports")
    md_path = export_markdown(report_md, out_dir / "ch15_agent_demo_report.md")

    print(
        json.dumps(
            {
                "status": out["status"],
                "review_passed": out["review_result"].get("passed"),
                "retries": out["retries"],
                "log_rounds": len(out["orchestration_log"]),
                "report_path": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
