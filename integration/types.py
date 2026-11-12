from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    order_id: str
    zone: int
    window_start: int
    window_end: int
    demand: int = 1
    service_minutes: float = 0.0
    synthetic: bool = True


@dataclass(frozen=True)
class Van:
    van_id: str
    capacity: int
