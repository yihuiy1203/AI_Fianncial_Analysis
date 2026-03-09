from __future__ import annotations

import json
from pathlib import Path

from ifa.indicators.dashboard import build_full_panel, export_to_excel


def main() -> None:
    panel = build_full_panel("000001", 2019, 2022)
    Path("data/features").mkdir(parents=True, exist_ok=True)
    panel.to_csv("data/features/full_panel_ch05.csv", index=False)
    export_to_excel(panel, Path("data/features/full_panel_ch05.xlsx"))
    print(
        json.dumps(
            {
                "rows": int(len(panel)),
                "csv": "data/features/full_panel_ch05.csv",
                "excel": "data/features/full_panel_ch05.xlsx",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
