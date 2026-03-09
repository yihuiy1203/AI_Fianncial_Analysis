from __future__ import annotations

import json
from pathlib import Path

from ifa.indicators.balance_sheet import panel_to_json_summary, run_balance_sheet_pipeline


def main() -> None:
    panel = run_balance_sheet_pipeline(
        code="000001",
        start_year=2019,
        end_year=2021,
        output_path=Path("data/features/ch03_balance_sheet_panel.csv"),
    )
    print(panel_to_json_summary(panel))
    print(json.dumps({"saved_to": "data/features/ch03_balance_sheet_panel.csv"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
