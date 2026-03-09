from __future__ import annotations

import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Normalize noisy report text into stable paragraphs."""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[\t\u00a0]+", " ", out)
    out = re.sub(r"[ ]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _split_paragraphs(text: str) -> list[str]:
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    return chunks


def _parse_text_fallback(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    normalized = clean_text(raw)
    paragraphs = [
        {"page_no": 1, "source_id": i, "text": p}
        for i, p in enumerate(_split_paragraphs(normalized), start=1)
    ]
    return {
        "paragraphs": paragraphs,
        "tables": [],
        "meta": {"file": str(path), "parser": "text_fallback"},
    }


def parse_pdf(pdf_path: str | Path) -> dict[str, object]:
    """
    Parse report file into standardized blocks.

    Notes:
    - If pypdf is available and path suffix is .pdf, it will parse PDF pages.
    - Otherwise, it falls back to UTF-8 text parsing for teaching/demo reproducibility.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            paragraphs: list[dict[str, object]] = []
            for page_idx, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                normalized = clean_text(raw)
                for offset, p in enumerate(_split_paragraphs(normalized), start=1):
                    paragraphs.append(
                        {
                            "page_no": page_idx,
                            "source_id": f"{page_idx}-{offset}",
                            "text": p,
                        }
                    )
            return {
                "paragraphs": paragraphs,
                "tables": [],
                "meta": {"file": str(path), "parser": "pypdf"},
            }
        except Exception:
            return _parse_text_fallback(path)

    return _parse_text_fallback(path)
