"""
Load / demand specialist agent.

Classifies instantaneous consumption, evaluates peak-demand (TOU / transformer
rating) risk and flags spikes against the 1-hour forecast.

Bug fixed
---------
``DemandAgent`` was constructed with the **whole** ``limits.yaml`` mapping but
then read ``config.get("peak_threshold_kw")`` at the top level, while
``limits.yaml`` had no ``demand:`` section at all. The configured threshold was
therefore unreachable and the agent always fell back to the hard-coded ``6.0``
default. A ``demand:`` section now exists in ``config/limits.yaml`` and the
agent receives its own sub-block, exactly like ``BatteryAgent`` does.
"""

from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent


class DemandAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DemandAgent", config)
        self.peak_threshold_kw = float(self.cfg("peak_threshold_kw", 6.0))
        self.moderate_threshold_kw = float(self.cfg("moderate_threshold_kw", 3.0))
        self.spike_delta_kw = float(self.cfg("spike_delta_kw", 2.0))

    def report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        load_now = max(0.0, self.num(state, "load_kw", 0.0))
        load_fc_1h = self.num(state, "load_fc_1h", load_now)
        has_fc_4h = state.get("load_fc_4h") is not None
        load_fc_4h = self.num(state, "load_fc_4h", load_now)

        peak_risk = bool(
            load_now >= self.peak_threshold_kw or load_fc_1h >= self.peak_threshold_kw
        )

        if load_now > self.peak_threshold_kw:
            load_category = "critical_high"
        elif load_now > self.moderate_threshold_kw:
            load_category = "moderate"
        else:
            load_category = "low"

        # A spike is load that is materially above what the 1 h forecast expects.
        spike_kw = max(0.0, load_now - load_fc_1h)
        spike_detected = bool(spike_kw >= self.spike_delta_kw)

        return {
            "agent": self.name,
            "load_now_kw": round(load_now, 3),
            "load_fc_1h_kw": round(load_fc_1h, 3),
            "load_fc_4h_kw": round(load_fc_4h, 3),
            "load_fc_4h_available": bool(has_fc_4h),
            "load_category": load_category,
            "peak_threshold_kw": self.peak_threshold_kw,
            "peak_risk": peak_risk,
            "spike_detected": spike_detected,
            "spike_kw": round(spike_kw, 3),
            "estimated_daily_kwh": round(load_now * 24.0, 2),
        }
