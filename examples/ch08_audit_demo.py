from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from ifa.audit import extract_kam, extract_metadata, extract_opinion_type, parse_pdf, summarize_kam


def main() -> None:
    base = Path(__file__).resolve().parent / f"_tmp_ch08_demo_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=False)
    try:
        report_path = base / "000002_2023_audit_report.txt"
        report_path.write_text(
            """
            审计报告

            我们发表标准无保留意见。

            关键审计事项

            事项一 应收账款可回收性
            客户回款周期延长，坏账计提依赖管理层估计，存在判断不确定性。

            事项二 商誉减值
            并购标的盈利能力波动，减值测试关键假设变化会影响利润。

            其他信息
            管理层与治理层责任说明。
            """,
            encoding="utf-8",
        )

        parsed = parse_pdf(report_path)
        paragraphs = parsed["paragraphs"]
        opinion = extract_opinion_type(paragraphs)
        kam_items = extract_kam(paragraphs)
        meta = extract_metadata(paragraphs, file_name=report_path.name)

        rows = []
        for item in kam_items:
            summary = summarize_kam(item["kam_text"], llm_client=None)
            rows.append(
                {
                    "stock_code": meta["stock_code"],
                    "year": meta["year"],
                    "opinion_type": opinion["opinion_type"],
                    "kam_title": item["kam_title"],
                    "kam_text": item["kam_text"],
                    "topic": summary["topic"],
                    "risk_point": summary["risk_point"],
                    "audit_response": summary["audit_response"],
                    "one_line_summary": summary["one_line_summary"],
                }
            )

        audit_df = pd.DataFrame(rows)
        if not audit_df.empty:
            audit_df["year"] = pd.to_numeric(audit_df["year"], errors="coerce").astype("Int64")
        risk_df = pd.DataFrame(
            {
                "stock_code": ["000002"],
                "year": [2023],
                "risk_score": [0.47],
            }
        )
        if not audit_df.empty:
            merged = audit_df.merge(risk_df, on=["stock_code", "year"], how="left")
        else:
            merged = audit_df

        print(
            json.dumps(
                {
                    "parsed_paragraphs": len(paragraphs),
                    "opinion_type": opinion["opinion_type"],
                    "kam_count": len(kam_items),
                    "merged_rows": int(len(merged)),
                    "columns": merged.columns.tolist() if not merged.empty else [],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
