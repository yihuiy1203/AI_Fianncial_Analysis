from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_semver(version: str) -> tuple[int, int, int]:
    core = version.strip().lstrip("v").split("-")[0]
    parts = core.split(".")
    if len(parts) != 3:
        raise ValueError("version must be semantic version like v1.2.3")
    try:
        major, minor, patch = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise ValueError("version contains non-integer segments") from exc
    return major, minor, patch


def bump_version(version: str, part: str = "patch") -> str:
    major, minor, patch = parse_semver(version)
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError("part must be one of: major, minor, patch")
    return f"v{major}.{minor}.{patch}"


def build_release_checklist(
    version: str,
    test_summary: dict[str, Any],
    doc_paths: list[str | Path],
    artifacts: list[str | Path],
) -> list[dict[str, Any]]:
    parse_semver(version)
    docs_ok = all(Path(p).exists() for p in doc_paths)
    artifacts_ok = all(Path(p).exists() for p in artifacts)
    tests_ok = bool(test_summary.get("failed", 1) == 0)

    return [
        {"item": "tests_green", "passed": tests_ok, "detail": f"failed={test_summary.get('failed', 'n/a')}"},
        {"item": "docs_ready", "passed": docs_ok, "detail": f"count={len(doc_paths)}"},
        {"item": "artifacts_ready", "passed": artifacts_ok, "detail": f"count={len(artifacts)}"},
        {"item": "version_valid", "passed": True, "detail": version},
    ]


def validate_release_ready(checklist: list[dict[str, Any]]) -> dict[str, Any]:
    failed_items = [str(item.get("item")) for item in checklist if not bool(item.get("passed"))]
    return {"ready": len(failed_items) == 0, "failed_items": failed_items}


def export_release_manifest(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
