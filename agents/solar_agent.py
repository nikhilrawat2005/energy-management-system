"""
Solar generation specialist agent.

Evaluates current PV output, the expected surplus (now and one hour ahead) and
classifies the array's operating condition so the coordinator and the Grafana
audit page can explain *why* a decision was taken.

Improvements
------------
* All thresholds come from ``config/limits.yaml`` (``solar:`` block) instead of
  magic numbers.
* ``cloud_drop_alarm`` is only raised when the *pyranometer* sees irradiance but
  the inverter is not delivering. The previous check flagged "cloud drop" for
  every low-PV reading between 06:00 and 18:59, which is simply what a PV
  string looks like at sunrise and sunset.
* ``solar_fc_4h_kw`` used to silently default to ``0.0`` when the 4 h forecast
  was absent, which is indistinguishable from "no sun". The value is now marked
  as unavailable.
"""

from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent


class SolarAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SolarAgent", config)
        solar_cfg = (config or {}).get("solar", {}) or {}
        self.peak_production_kw = float(solar_cfg.get("peak_production_kw", 7.0))
        self.daytime_hours = set(
            int(h) for h in (solar_cfg.get("daytime_hours", list(range(6, 19))) or [])
        )
        # Below this irradiance the array simply cannot be producing.
        self.min_expected_irradiance = float(solar_cfg.get("min_expected_irradiance", 120.0))
        # System size used to convert irradiance into an expected kW.
        self.peak_kw_at_1000_wm2 = float(solar_cfg.get("peak_kw_at_1000_wm2", 8.5))

    def report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        solar_now = max(0.0, self.num(state, "solar_kw", 0.0))
        solar_fc_1h = self.num(state, "solar_fc_1h", solar_now)
        has_fc_4h = state.get("solar_fc_4h") is not None
        solar_fc_4h = self.num(state, "solar_fc_4h", solar_now)
        load_now = max(0.0, self.num(state, "load_kw", 0.0))
        load_fc_1h = self.num(state, "load_fc_1h", load_now)
        irradiance = max(0.0, self.num(state, "irradiance_wm2", 0.0))
        hour = self.num(state, "hour", 0.0)

        current_surplus = max(0.0, solar_now - load_now)
        expected_surplus_1h = max(0.0, solar_fc_1h - load_fc_1h)

        # Expected output implied by the pyranometer, allowing for system losses.
        expected_kw = irradiance / 1000.0 * self.peak_kw_at_1000_wm2
        underperformance_pct = 0.0
        if expected_kw > 0.5:
            underperformance_pct = max(0.0, (expected_kw - solar_now) / expected_kw * 100.0)

        hour_int = int(hour) % 24
        status = "normal"
        if solar_now >= self.peak_production_kw:
            status = "peak_production"
        elif (
            hour_int in self.daytime_hours
            and irradiance >= self.min_expected_irradiance
            and solar_now < 0.1
        ):
            status = "cloud_drop_alarm"
        elif hour_int not in self.daytime_hours and solar_now < 0.1:
            status = "night_idle"
        elif underperformance_pct >= 60.0:
            status = "underperformance"

        return {
            "agent": self.name,
            "solar_now_kw": round(solar_now, 3),
            "solar_fc_1h_kw": round(solar_fc_1h, 3),
            "solar_fc_4h_kw": round(solar_fc_4h, 3),
            "solar_fc_4h_available": bool(has_fc_4h),
            "irradiance_wm2": round(irradiance, 1),
            "expected_kw_from_irradiance": round(expected_kw, 3),
            "underperformance_pct": round(underperformance_pct, 1),
            "current_surplus_kw": round(current_surplus, 3),
            "expected_surplus_1h_kw": round(expected_surplus_1h, 3),
            "status": status,
        }
