from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Timer:
    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start


@dataclass
class StageTimes:
    values: Dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + float(value)

    def get(self, name: str) -> float:
        return float(self.values.get(name, 0.0))
