from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_evidence_table(results: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if isinstance(results, list):
        for idx, item in enumerate(results, start=1):
            rows.append(
                {
                    "id": idx,
                    "source": str(item.get("source", item.get("tool", "unknown"))),
                    "claim": str(item.get("claim", item.get("summary", ""))),
                    "evidence": str(item.get("evidence", item.get("detail", ""))),
                }
            )
        return pd.DataFrame(rows)

    analysis = results.get("analysis_result", results)
    evidence = analysis.get("evidence", {}) if isinstance(analysis, dict) else {}
    for idx, (tool_name, output) in enumerate(evidence.items(), start=1):
        rows.append(
            {
                "id": idx,
                "source": tool_name,
                "claim": f"Output from {tool_name}",
                "evidence": str(output),
            }
        )

    return pd.DataFrame(rows)


def _table_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| id | source | claim | evidence |\n|---|---|---|---|\n| - | - | - | - |"

    cols = ["id", "source", "claim", "evidence"]
    use = [c for c in cols if c in df.columns]
    header = "| " + " | ".join(use) + " |"
    sep = "|" + "|".join(["---"] * len(use)) + "|"
    lines = [header, sep]
    for _, row in df[use].iterrows():
        values = [str(row[c]).replace("\n", " ").replace("|", "\\|") for c in use]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def embed_figures(report_text: str, figure_paths: list[str | Path]) -> str:
    if not figure_paths:
        return report_text
    lines = ["", "## Figures"]
    for p in figure_paths:
        path = Path(p)
        lines.append(f"- ![{path.stem}]({path.as_posix()})")
    return report_text + "\n" + "\n".join(lines) + "\n"


def build_report(results: dict[str, Any], meta: dict[str, Any] | None = None, template_path: str | Path | None = None) -> str:
    metadata = dict(meta or {})
    if template_path:
        metadata["template_path"] = str(template_path)

    analysis = results.get("analysis_result", {})
    review = results.get("review_result", {})

    summary = str(analysis.get("summary", ""))
    status = str(results.get("status", review.get("passed", "unknown")))
    review_summary = str(review.get("summary", ""))
    warnings = analysis.get("warnings", []) if isinstance(analysis, dict) else []

    evidence_df = build_evidence_table(results)
    evidence_md = _table_to_markdown(evidence_df)

    meta_lines = []
    for k, v in sorted(metadata.items()):
        meta_lines.append(f"- {k}: {v}")

    warning_text = "无" if not warnings else "; ".join(str(w) for w in warnings)

    sections = [
        "# Automated Financial Analysis Report",
        "",
        "## Metadata",
        *meta_lines,
        "",
        "## Executive Summary",
        summary or "No summary available.",
        "",
        "## Review",
        f"- status: {status}",
        f"- reviewer_note: {review_summary or 'n/a'}",
        f"- warnings: {warning_text}",
        "",
        "## Evidence Map",
        evidence_md,
        "",
        "## Boundaries",
        "- This report is auto-generated from tool outputs.",
        "- High-risk decisions require human review.",
        "",
    ]
    return "\n".join(sections)


def export_markdown(report_md: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_md, encoding="utf-8")
    return path


def _escape_pdf_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def export_pdf(report_md: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Minimal single-page PDF writer for ASCII/UTF-8 markdown text.
    lines = [ln[:100] for ln in report_md.splitlines()[:45]]
    if not lines:
        lines = ["(empty report)"]

    text_cmds = []
    y = 780
    for line in lines:
        safe = _escape_pdf_text(line)
        text_cmds.append(f"1 0 0 1 40 {y} Tm ({safe}) Tj")
        y -= 16
    stream = "BT /F1 10 Tf " + " ".join(text_cmds) + " ET"

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream.encode('latin-1', errors='replace'))} >>\nstream\n{stream}\nendstream",
    ]

    pdf = "%PDF-1.4\n"
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf.encode("latin-1", errors="replace")))
        pdf += f"{i} 0 obj\n{obj}\nendobj\n"

    xref_start = len(pdf.encode("latin-1", errors="replace"))
    pdf += f"xref\n0 {len(objects) + 1}\n"
    pdf += "0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n"
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"

    path.write_bytes(pdf.encode("latin-1", errors="replace"))
    return path
