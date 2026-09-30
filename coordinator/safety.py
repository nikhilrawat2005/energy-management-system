"""
Mandatory hardware & operational safety interlock for the MAEMS coordinator.

Every proposed action - whether it came from the RL policy or the deterministic
baseline rules - passes through :meth:`SafetyLayer.validate_and_override` before
it can reach the inverters. The layer is deliberately *independent* of the
decision logic so a buggy or badly-trained policy can never drive the pack
outside its envelope.

Guards implemented (in priority order)
--------------------------------------
1. **Action validation** - unknown action ids are rejected, never raised on.
2. **Critical thermal lockout** - battery power flow is cut above
   ``battery.temp_critical_c``.
3. **Grid blackout / islanding** - grid charging is blocked, and the battery is
   forced to carry the load when the utility drops.
4. **Grid-quality interlock** - voltage outside +/-``grid.voltage_tolerance_pct``
   of nominal, or frequency outside +/-``grid.freq_tolerance_hz``, is treated as
   a loss of grid even when the utility still reports "connected" (brown-outs and
   diesel gensets report exactly that).
5. **Import limit** - blocks grid charging that would push the point of common
   coupling above ``grid.max_import_kw``.
6. **Overcharge prevention** - no charging at or above ``battery.soc_max``.
7. **Overdischarge protection** - no discharging at or below ``battery.soc_min``.
8. **Relay rate limit** - battery power-flow actions cannot *change direction*
   more often than ``safety.rate_limit_seconds``, protecting inverters and
   contactors from chatter.

Fixes vs. the previous implementation
-------------------------------------
* The rate limiter was described in the class docstring and its parameters were
  read from config, but **no code ever used it**. It is now enforced.
* The ``grid:`` block of ``config/limits.yaml`` (voltage/frequency tolerance,
  ``max_import_kw``) was **entirely unused**; it now drives real guards.
* ``soc``/``temperature``/``grid_status`` were coerced without checking for
  ``None`` or non-finite values, which could raise inside the control loop.
* Returning the previous action after a rate-limit trip is avoided; the layer
  degrades to a safe idle (action 0) instead of replaying a stale decision.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

# Battery power-flow actions are the ones that move a relay/contactor.
_POWER_FLOW_ACTIONS = (1, 2, 3)


def _as_float(value: Any, fallback: float) -> float:
    """Best-effort float coercion that never raises inside the control loop."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(result) or math.isinf(result):
        return fallback
    return result


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


class SafetyLayer:
    def __init__(self, limits_cfg: Dict[str, Any]):
        batt_cfg = limits_cfg.get("battery", {}) or {}
        self.soc_min = float(batt_cfg.get("soc_min", 0.10))
        self.soc_max = float(batt_cfg.get("soc_max", 0.95))
        self.temp_max_c = float(batt_cfg.get("temp_max_c", 45.0))
        self.temp_critical_c = float(batt_cfg.get("temp_critical_c", 55.0))
        self.max_charge_kw = float(batt_cfg.get("max_charge_kw", 5.0))
        self.max_discharge_kw = float(batt_cfg.get("max_discharge_kw", 5.0))

        grid_cfg = limits_cfg.get("grid", {}) or {}
        self.voltage_nominal = float(grid_cfg.get("voltage_nominal", 230.0))
        self.voltage_tolerance_pct = float(grid_cfg.get("voltage_tolerance_pct", 10.0))
        self.freq_nominal = float(grid_cfg.get("freq_nominal", 50.0))
        self.freq_tolerance_hz = float(grid_cfg.get("freq_tolerance_hz", 1.5))
        self.max_import_kw = float(grid_cfg.get("max_import_kw", 25.0))
        self.v_min = self.voltage_nominal * (1.0 - self.voltage_tolerance_pct / 100.0)
        self.v_max = self.voltage_nominal * (1.0 + self.voltage_tolerance_pct / 100.0)
        self.f_min = self.freq_nominal - self.freq_tolerance_hz
        self.f_max = self.freq_nominal + self.freq_tolerance_hz

        safety_cfg = limits_cfg.get("safety", {}) or {}
        self.rate_limit_seconds = float(safety_cfg.get("rate_limit_seconds", 900))
        self.enforce_rate_limit = bool(safety_cfg.get("enforce_rate_limit", True))
        self.enforce_grid_quality = bool(safety_cfg.get("enforce_grid_quality", True))
        self.island_mode_on_grid_loss = bool(
            safety_cfg.get("island_mode_on_grid_loss", True)
        )

        # Relay / inverter protection state.
        self.last_action: Optional[int] = None
        self.last_action_time: float = 0.0
        self.trip_count: int = 0
        self.last_trip_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the relay-protection state (used between evaluation runs)."""
        self.last_action = None
        self.last_action_time = 0.0
        self.trip_count = 0
        self.last_trip_reason = None

    def _trip(self, reason: str) -> None:
        self.trip_count += 1
        self.last_trip_reason = reason

    def grid_quality_ok(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """Return ``(ok, reason)`` for the measured grid voltage / frequency."""
        voltage = _as_float(state.get("grid_voltage", self.voltage_nominal), self.voltage_nominal)
        freq = _as_float(state.get("grid_freq", self.freq_nominal), self.freq_nominal)
        if not (self.v_min <= voltage <= self.v_max):
            return False, (
                f"grid voltage {voltage:.1f}V outside "
                f"{self.v_min:.1f}-{self.v_max:.1f}V band"
            )
        if not (self.f_min <= freq <= self.f_max):
            return False, (
                f"grid frequency {freq:.2f}Hz outside "
                f"{self.f_min:.2f}-{self.f_max:.2f}Hz band"
            )
        return True, "grid quality within tolerance"

    # ------------------------------------------------------------------
    # Main interlock
    # ------------------------------------------------------------------
    def validate_and_override(
        self, proposed_action: Any, state: Dict[str, Any], current_time_sec: float
    ) -> Tuple[int, str]:
        """Validate a proposed action and return the action that may be executed.

        Actions:
          0: IDLE / SOLAR_FIRST  (solar serves load, grid covers the balance)
          1: CHARGE_SOLAR       (surplus solar into the battery)
          2: DISCHARGE_BATT     (battery serves load)
          3: CHARGE_GRID        (off-peak grid energy into the battery)
        """
        state = state or {}

        # 0. Action validation - a malformed policy output must never crash the loop.
        try:
            proposed_action = int(proposed_action)
        except (TypeError, ValueError):
            self._trip("invalid action id")
            return 0, "SAFETY_OVERRIDE: Unrecognised action id - defaulting to safe idle."

        if proposed_action not in (0, 1, 2, 3):
            self._trip("action out of range")
            return 0, (
                f"SAFETY_OVERRIDE: Action {proposed_action} is out of range (0-3) - "
                "defaulting to safe idle."
            )

        soc = _as_float(state.get("soc", 0.5), 0.5)
        soc = min(max(soc, 0.0), 1.0)
        temp = _as_float(state.get("batt_temp", 25.0), 25.0)
        grid_status = _as_int(state.get("grid_status", 1), 1)
        load_kw = _as_float(state.get("load_kw", 0.0), 0.0)

        # 1. Critical thermal lockout ------------------------------------
        if temp >= self.temp_critical_c:
            if proposed_action in _POWER_FLOW_ACTIONS:
                self._trip("critical thermal")
                return 0, (
                    f"SAFETY_OVERRIDE: Battery Temp ({temp:.1f}C) >= Critical limit "
                    f"({self.temp_critical_c}C). Battery power flow locked out."
                )

        # 2. Grid quality interlock (brown-out / generator / bad sensor) --
        quality_ok, quality_reason = self.grid_quality_ok(state)
        if self.enforce_grid_quality and grid_status == 1 and not quality_ok:
            # The inverter cannot sync to an out-of-band grid: behave as an outage.
            grid_status = 0

        if grid_status == 0:
            if proposed_action == 3:
                self._trip("blackout grid charge")
                return 0, "SAFETY_OVERRIDE: Grid Blackout detected! Grid charging disabled."
            if (
                self.island_mode_on_grid_loss
                and proposed_action == 0
                and soc > self.soc_min
            ):
                self._trip("island mode")
                return 2, (
                    "SAFETY_ISLAND_MODE: Grid outage"
                    + (f" ({quality_reason})" if not quality_ok else "")
                    + "! Forcing Battery discharge to sustain microgrid load."
                )

        # 3. Point-of-common-coupling import limit ----------------------
        if proposed_action == 3:
            projected_import = max(0.0, load_kw) + self.max_charge_kw
            if projected_import > self.max_import_kw:
                self._trip("import limit")
                return 0, (
                    f"SAFETY_OVERRIDE: Projected grid import {projected_import:.1f}kW "
                    f"exceeds limit {self.max_import_kw:.1f}kW. Grid charging blocked."
                )

        # 4. Overcharge prevention --------------------------------------
        if soc >= self.soc_max and proposed_action in (1, 3):
            self._trip("soc ceiling")
            return 0, (
                f"SAFETY_OVERRIDE: Battery SoC ({soc * 100:.1f}%) >= Max limit "
                f"({self.soc_max * 100:.1f}%). Charging blocked."
            )

        # 5. Overdischarge protection -----------------------------------
        if soc <= self.soc_min and proposed_action == 2:
            self._trip("soc floor")
            return 0, (
                f"SAFETY_OVERRIDE: Battery SoC ({soc * 100:.1f}%) <= Min reserve "
                f"({self.soc_min * 100:.1f}%). Discharge blocked."
            )

        # 6. Relay rate limit (contactor / inverter protection) ----------
        if self.enforce_rate_limit and self._is_rate_limited(proposed_action, current_time_sec):
            self._trip("rate limit")
            return 0, (
                f"SAFETY_RATE_LIMIT: Power-flow change blocked, previous action held for "
                f"{self.rate_limit_seconds:.0f}s to protect the inverter relay. "
                "Falling back to safe idle."
            )

        # 7. Action confirmed safe --------------------------------------
        if proposed_action in _POWER_FLOW_ACTIONS:
            self.last_action = proposed_action
            self.last_action_time = _as_float(current_time_sec, 0.0)
        elif proposed_action == 0 and self.last_action is None:
            self.last_action = 0
            self.last_action_time = _as_float(current_time_sec, 0.0)

        if temp >= self.temp_max_c and proposed_action in (1, 3):
            return proposed_action, (
                f"SAFE_THROTTLED: Battery Temp ({temp:.1f}C) >= throttling limit "
                f"({self.temp_max_c}C). Charge power derated by the battery agent."
            )
        return proposed_action, "SAFE_CONFIRMED"

    # ------------------------------------------------------------------
    def _is_rate_limited(self, proposed_action: int, current_time_sec: Any) -> bool:
        """True when a battery power-flow change is too soon after the last one."""
        if self.rate_limit_seconds <= 0:
            return False
        if proposed_action not in _POWER_FLOW_ACTIONS:
            return False
        if self.last_action is None or self.last_action not in _POWER_FLOW_ACTIONS:
            return False
        if proposed_action == self.last_action:
            return False  # holding steady is exactly what we want
        now = _as_float(current_time_sec, self.last_action_time)
        if now < self.last_action_time:
            return False  # clock stepped backwards (replay); do not lock out
        return (now - self.last_action_time) < self.rate_limit_seconds
