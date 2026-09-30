"""
Electricity tariff / price specialist agent.

Classifies the live ToD tariff, identifies economic windows for grid charging,
and cross-checks the billed price against the configured time-of-use schedule.

Bug fixed
---------
The ``tariff:`` block of ``config/limits.yaml`` (off-peak 4.5 / normal 7.0 /
peak 11.5 Rs/kWh plus the hour lists) was **never read by any module** - it was
pure decoration. ``PriceAgent`` instead hard-coded ``<= 5.0`` and ``>= 10.0``,
so a tariff change in the YAML had no effect and the classifier disagreed with
the configured rates. The tier boundaries are now derived from the config, and
the observed price is validated against the published schedule, producing a
``tariff_mismatch`` flag that the audit dashboard can surface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.base import BaseAgent


class PriceAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PriceAgent", config)
        tariff = (config or {}).get("tariff", {}) or {}
        self.currency = tariff.get("currency", "INR")
        self.off_peak_rate = float(tariff.get("off_peak", {}).get("rate_per_kwh", 4.5))
        self.normal_rate = float(tariff.get("normal", {}).get("rate_per_kwh", 7.0))
        self.peak_rate = float(tariff.get("peak", {}).get("rate_per_kwh", 11.5))
        self.off_peak_hours = set(tariff.get("off_peak", {}).get("hours", []) or [])
        self.normal_hours = set(tariff.get("normal", {}).get("hours", []) or [])
        self.peak_hours = set(tariff.get("peak", {}).get("hours", []) or [])

    # ------------------------------------------------------------------
    def expected_rate_for_hour(self, hour: float) -> Optional[float]:
        """Rate the published ToD schedule predicts for *hour*, or ``None``."""
        hour_int = int(hour) % 24
        if hour_int in self.off_peak_hours:
            return self.off_peak_rate
        if hour_int in self.peak_hours:
            return self.peak_rate
        if hour_int in self.normal_hours:
            return self.normal_rate
        return None

    def classify(self, price: float) -> str:
        """Map a live price to a tier using the configured rate boundaries."""
        if price <= self.off_peak_rate:
            return "OFF_PEAK_CHEAP"
        if price >= self.peak_rate:
            return "PEAK_EXPENSIVE"
        return "NORMAL"

    # ------------------------------------------------------------------
    def report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        price = self.num(state, "price", self.normal_rate)
        grid_status = self.integer(state, "grid_status", 1)
        hour = self.num(state, "hour", 12.0)

        price_tier = self.classify(price)
        cheap_window = price_tier == "OFF_PEAK_CHEAP"
        grid_ok = bool(grid_status == 1)

        expected_rate = self.expected_rate_for_hour(hour)
        if expected_rate is None:
            tariff_mismatch = False
        else:
            tariff_mismatch = abs(expected_rate - price) > 0.01

        return {
            "agent": self.name,
            "price_inr_per_kwh": round(price, 2),
            "price_tier": price_tier,
            "cheap_window": cheap_window,
            "peak_window": price_tier == "PEAK_EXPENSIVE",
            "grid_ok": grid_ok,
            "grid_voltage_v": round(self.num(state, "grid_voltage", 230.0), 1),
            "grid_freq_hz": round(self.num(state, "grid_freq", 50.0), 2),
            "expected_rate_for_hour": expected_rate,
            "tariff_mismatch": bool(tariff_mismatch),
            "off_peak_rate": self.off_peak_rate,
            "normal_rate": self.normal_rate,
            "peak_rate": self.peak_rate,
        }
