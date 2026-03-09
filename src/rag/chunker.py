from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    year: int | None
    section: str
    start_offset: int
    end_offset: int


def normalize_text(text: str) -> str:
    out = str(text).replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def chunk_by_length(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >=0 and smaller than chunk_size")

    norm = normalize_text(text)
    if not norm:
        return []

    step = chunk_size - overlap
    out: list[str] = []
    i = 0
    while i < len(norm):
        part = norm[i : i + chunk_size].strip()
        if part:
            out.append(part)
        i += step
    return out


def chunk_by_semantic(text: str, delimiters: list[str] | None = None, min_len: int = 0) -> list[str]:
    norm = normalize_text(text)
    if not norm:
        return []

    if delimiters is None:
        delimiters = ["\n\n", "。", "；", "!", "?", "！", "？"]

    # First split by paragraph-level delimiter; then split over-long parts by sentence delimiters.
    para_delim = delimiters[0]
    sent_delims = delimiters[1:]
    parts = [x.strip() for x in norm.split(para_delim) if x.strip()]
    if not parts:
        parts = [norm]

    refined: list[str] = []
    for p in parts:
        if len(p) <= max(min_len * 2, 120):
            refined.append(p)
            continue
        segs = [p]
        for d in sent_delims:
            tmp: list[str] = []
            for s in segs:
                tmp.extend([x.strip() for x in s.split(d) if x.strip()])
            segs = tmp
        refined.extend(segs)
    parts = refined

    # Optionally merge very short segments with neighbors.
    merged: list[str] = []
    for seg in parts:
        if not merged:
            merged.append(seg)
            continue
        # Merge only when both sides are short; keep normal short sentences as independent chunks.
        if min_len > 0 and len(seg) < min_len and len(merged[-1]) < min_len:
            merged[-1] = (merged[-1] + " " + seg).strip()
        else:
            merged.append(seg)
    return [m for m in merged if m.strip()]


def add_metadata(
    chunks: list[str],
    source: str,
    year: int | None = None,
    section: str = "",
    base_id: str = "chunk",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pos = 0
    for i, text in enumerate(chunks, start=1):
        start = pos
        end = pos + len(text)
        cid = f"{base_id}_{i}"
        out.append(
            {
                "chunk_id": cid,
                "text": text,
                "source": source,
                "year": year,
                "section": section,
                "start_offset": start,
                "end_offset": end,
            }
        )
        pos = end + 1
    return out


def deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out: list[dict[str, Any]] = []
    for c in chunks:
        norm = normalize_text(c.get("text", ""))
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(c)
    return out
