from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMetric(ABC):
    name: str

    @abstractmethod
    def __call__(self, prediction, target) -> float:
        raise NotImplementedError
