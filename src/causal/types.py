from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalResult:
    method: str
    effect: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    note: str = ""
