from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable


class MetricTracker:
    """Keeps running sums/averages for metrics and losses."""

    def __init__(self, *keys: Iterable[str]) -> None:
        self.data = {key: {"sum": 0.0, "count": 0} for key in keys}

    def update(self, key: str, value: float, n: int = 1) -> None:
        if key not in self.data:
            self.data[key] = {"sum": 0.0, "count": 0}
        self.data[key]["sum"] += value * n
        self.data[key]["count"] += n

    def avg(self, key: str) -> float:
        entry = self.data.get(key, {"sum": 0.0, "count": 0})
        if entry["count"] == 0:
            return 0.0
        return entry["sum"] / entry["count"]

    def to_dict(self) -> Dict[str, float]:
        return {k: self.avg(k) for k in self.data}

    def reset(self) -> None:
        for key in self.data:
            self.data[key]["sum"] = 0.0
            self.data[key]["count"] = 0
