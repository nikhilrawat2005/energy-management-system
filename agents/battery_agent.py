"""
Battery Management System (BMS) specialist agent.

Reports state of charge, pack health, thermal status and - crucially - the
*power head-room* currently available for charging and discharging, in kW.

Bug fixed
---------
The thermal-throttling branch previously computed::

    avail_charge = min(max_charge_kw * 0.5, (soc_max - soc) * capacity_kwh)

``(soc_max - soc) * capacity_kwh`` is an **energy** quantity (kWh) because
``soc_max - soc`` is a dimensionless fraction, but it was then published as
``avail_charge_kw`` (kW). A 3.75 kWh head-room was reported as 3.75 kW, which is
not the same thing once you integrate over a 15-minute control step (3.75 kW x
0.25 h = 0.94 kWh actually absorbed). The energy head-room is now converted to a
power with the configured control interval, and both quantities are reported so
downstream consumers can choose.
"""

from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent, clamp


class BatteryAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any], control_interval_h: float = 0.25):
        super().__init__("BatteryAgent", config)
        self.capacity_kwh = float(self.cfg("capacity_kwh", 15.0))
        self.soc_min = float(self.cfg("soc_min", 0.10))
        self.soc_max = float(self.cfg("soc_max", 0.95))
        self.max_charge_kw = float(self.cfg("max_charge_kw", 5.0))
        self.max_discharge_kw = float(self.cfg("max_discharge_kw", 5.0))
        self.temp_max_c = float(self.cfg("temp_max_c", 45.0))
        self.temp_critical_c = float(self.cfg("temp_critical_c", 55.0))
        # Derate factor applied above temp_max_c (charge is the stressful direction).
        self.thermal_derate = float(self.cfg("thermal_derate", 0.5))
        self.control_interval_h = max(1e-6, float(control_interval_h))

    def report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        soc = clamp(self.num(state, "soc", 0.5), 0.0, 1.0)
        temp = self.num(state, "batt_temp", 27.5)

        is_critical_hot = temp >= self.temp_critical_c
        is_throttled = temp >= self.temp_max_c

        # Energy head-room (kWh) available before hitting a SoC boundary.
        charge_headroom_kwh = max(0.0, (self.soc_max - soc) * self.capacity_kwh)
        discharge_headroom_kwh = max(0.0, (soc - self.soc_min) * self.capacity_kwh)

        # Convert energy head-room to the power that consumes it in one control step.
        charge_limit_kw = charge_headroom_kwh / self.control_interval_h
        discharge_limit_kw = discharge_headroom_kwh / self.control_interval_h

        if is_critical_hot:
            health_flag = "CRITICAL_OVERHEAT_LOCKOUT"
            derate = 0.0
        elif is_throttled:
            health_flag = "THERMAL_THROTTLING"
            # Severity grows linearly from temp_max_c to temp_critical_c so the
            # derate is smooth rather than a step change.
            span = max(1e-6, self.temp_critical_c - self.temp_max_c)
            severity = clamp((temp - self.temp_max_c) / span, 0.0, 1.0)
            derate = max(0.0, 1.0 - severity) * max(self.thermal_derate, 0.0)
        else:
            health_flag = "HEALTHY"
            derate = 1.0

        avail_charge_kw = min(
            self.max_charge_kw * derate,
            self.max_charge_kw,
            charge_limit_kw,
        )
        avail_discharge_kw = min(
            self.max_discharge_kw * derate,
            self.max_discharge_kw,
            discharge_limit_kw,
        )

        return {
            "agent": self.name,
            "soc": round(soc, 4),
            "soc_pct": round(soc * 100.0, 1),
            "batt_temp_c": round(temp, 2),
            "capacity_kwh": self.capacity_kwh,
            "stored_energy_kwh": round(soc * self.capacity_kwh, 3),
            "headroom_charge_kwh": round(charge_headroom_kwh, 3),
            "headroom_discharge_kwh": round(discharge_headroom_kwh, 3),
            "avail_charge_kw": round(max(0.0, avail_charge_kw), 3),
            "avail_discharge_kw": round(max(0.0, avail_discharge_kw), 3),
            "max_charge_kw": self.max_charge_kw,
            "max_discharge_kw": self.max_discharge_kw,
            "thermal_derate": round(derate, 3),
            "health_flag": health_flag,
        }
