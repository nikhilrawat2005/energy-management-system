"""
Common base class and coercion helpers for all MAEMS specialist agents.

Every agent receives the same flat telemetry dictionary and returns a flat,
JSON-serialisable report. The helpers here make that contract defensive: a
malformed field coming off a Modbus register or a REST payload must degrade to
a safe default instead of raising inside the 15-minute control loop.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict

__all__ = ["BaseAgent", "as_float", "as_int", "clamp"]


def as_float(value: Any, fallback: float = 0.0) -> float:
    """Coerce *value* to a finite float, returning *fallback* if impossible.

    Guards against ``None``, empty strings, ``"n/a"``, ``NaN`` and ``Infinity``
    - all of which can reach the agent from a sensor bus.
    """
    if value is None:
        return fallback
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return fallback
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(result) or math.isinf(result):
        return fallback
    return result


def as_int(value: Any, fallback: int = 0) -> int:
    """Coerce *value* to an int. Accepts ``"1"``, ``1.0`` and ``"1.0"``."""
    return int(as_float(value, float(fallback)))


def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* into ``[low, high]`` (a no-op if ``low > high``)."""
    if low > high:
        low, high = high, low
    return max(low, min(high, value))


class BaseAgent(ABC):
    """Abstract specialist agent.

    Subclasses implement :meth:`report`, returning a dictionary that always
    contains at least an ``"agent"`` key so the coordinator can build a
    homogeneous ``agent_reports`` mapping for the API, the dashboards and the
    evaluation harness.
    """

    def __init__(self, name: str, config: Dict[str, Any] | None = None):
        self.name = name
        self.config: Dict[str, Any] = dict(config or {})

    @abstractmethod
    def report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Turn a telemetry snapshot into this agent's structured opinion."""
        raise NotImplementedError

    # -- convenience helpers available to every subclass -----------------
    @staticmethod
    def num(state: Dict[str, Any], key: str, fallback: float = 0.0) -> float:
        return as_float(state.get(key), fallback)

    @staticmethod
    def integer(state: Dict[str, Any], key: str, fallback: int = 0) -> int:
        return as_int(state.get(key), fallback)

    def cfg(self, key: str, fallback: Any = None) -> Any:
        """Read a value from this agent's own config block."""
        return self.config.get(key, fallback)
