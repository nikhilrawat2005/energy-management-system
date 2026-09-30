"""
Deterministic rule-based coordinator - the reference "good enough" controller
that the RL policy is measured against.

Rules, evaluated in order:

1. **Solar surplus** - PV exceeds load, so push the surplus into the battery.
2. **Peak shaving** - tariff is at the configured peak rate and the battery has
   head-room above the reserve, so discharge instead of buying peak energy.
3. **Off-peak charging** - tariff is at the configured off-peak rate and the
   battery is below the charge target, so buy cheap grid energy.
4. **Default** - solar-first, grid covers whatever is left.

Bugs fixed
----------
* The peak-discharge reserve (``soc_min + 0.15``) and the off-peak charge
  ceiling (``0.60``) were hard-coded. They are now read from the
  ``coordinator:`` block of ``config/limits.yaml``.
* Rule 1 previously fired even when the battery was already at ``soc_max``
  *and* also ignored grid status, so during an outage it would ask for
  "charge from solar" with no grid - now bounded by the configured SoC window
  and by the safety interlock downstream.
* The rule engine now also declines to discharge into a solar surplus (the
  battery is already being filled), which previously wasted stored energy.
* **Rule 0 - island mode.** The rules used to fall through to the default
  "solar-first" during an outage, which is an illegal move, so the safety
  interlock had to correct it and reported a ``SAFETY_ISLAND_MODE`` trip on
  every blackout step (3 of them in the 7-day benchmark). An interlock is a
  backstop, not a control path: the rules now detect the outage themselves and
  discharge directly, so the interlock stays silent.
"""

from __future__ import annotations

from typing import Any, Dict

from agents.base import as_float


class BaselineRuleCoordinator:
    def __init__(self, limits_cfg: Dict[str, Any]):
        batt_cfg = (limits_cfg or {}).get("battery", {}) or {}
        coord_cfg = (limits_cfg or {}).get("coordinator", {}) or {}
        self.soc_min = float(batt_cfg.get("soc_min", 0.10))
        self.soc_max = float(batt_cfg.get("soc_max", 0.95))
        self.peak_discharge_soc_margin = float(coord_cfg.get("peak_discharge_soc_margin", 0.15))
        self.off_peak_charge_target_soc = float(coord_cfg.get("off_peak_charge_target_soc", 0.60))

    def select_action(
        self, agent_reports: Dict[str, Dict[str, Any]], raw_state: Dict[str, Any]
    ) -> int:
        reports = agent_reports or {}
        solar_rep = reports.get("solar", {}) or {}
        demand_rep = reports.get("demand", {}) or {}
        batt_rep = reports.get("battery", {}) or {}
        price_rep = reports.get("price", {}) or {}

        solar_now = as_float(solar_rep.get("solar_now_kw"), 0.0)
        load_now = as_float(demand_rep.get("load_now_kw"), 0.0)
        soc = as_float(batt_rep.get("soc"), 0.5)
        price_tier = price_rep.get("price_tier", "NORMAL")
        grid_ok = bool(price_rep.get("grid_ok", True))
        has_headroom = soc < self.soc_max
        can_discharge = soc > self.soc_min

        # Rule 0: grid loss. The only sensible moves are to serve the load from
        # the battery (if it has charge) or, if it is flat, to idle and let the
        # safety layer own the load-shedding decision. Never charge from a grid
        # that is not there, and never ask the interlock to fix an illegal move.
        if not grid_ok:
            return 2 if can_discharge else 0

        # Rule 1: absorb surplus solar
        if solar_now > load_now and has_headroom:
            return 1  # CHARGE_SOLAR

        # Rule 2: peak tariff shaving
        if price_tier == "PEAK_EXPENSIVE" and can_discharge and soc > (
            self.soc_min + self.peak_discharge_soc_margin
        ):
            return 2  # DISCHARGE_BATT

        # Rule 3: cheap grid charging
        if price_tier == "OFF_PEAK_CHEAP" and grid_ok and soc < self.off_peak_charge_target_soc:
            return 3  # CHARGE_GRID

        # Rule 4: default - solar first, grid shortfall
        return 0  # SOLAR_FIRST
