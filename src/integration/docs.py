from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any


def _iter_public_functions(module: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        if getattr(obj, "__module__", "") != module.__name__:
            continue
        doc = inspect.getdoc(obj) or ""
        summary = doc.splitlines()[0] if doc else ""
        rows.append((name, summary))
    rows.sort(key=lambda x: x[0])
    return rows


def generate_api_markdown(modules: list[str], title: str = "API Reference") -> str:
    lines = [f"# {title}", ""]
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        lines.append(f"## `{mod_name}`")
        funcs = _iter_public_functions(mod)
        if not funcs:
            lines.append("- (no public functions)")
            lines.append("")
            continue
        for fn_name, summary in funcs:
            text = summary if summary else "No description."
            lines.append(f"- `{fn_name}`: {text}")
        lines.append("")
    return "\n".join(lines)


def generate_quickstart(
    install_cmd: str = "pip install -e .",
    test_cmd: str = "pytest -q",
    demo_cmd: str = "python examples/ch16_integration_demo.py",
) -> str:
    return "\n".join(
        [
            "# Quickstart",
            "",
            "```bash",
            install_cmd,
            test_cmd,
            demo_cmd,
            "```",
            "",
        ]
    )


def export_markdown(content: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
