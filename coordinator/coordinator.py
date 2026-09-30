"""
Central multi-agent coordinator.

Responsibilities
----------------
1. Poll the four specialist agents (Solar, Demand, Battery, Price) for their
   structured opinions on the current telemetry snapshot.
2. Ask the selected decision engine (RL policy or deterministic rules) to
   propose a dispatch action.
3. Pass that proposal through the :class:`~coordinator.safety.SafetyLayer`
   interlock, which has the final say.

Bugs fixed
----------
* ``config/limits.yaml`` and ``models/rl_policy.json`` were opened through
  **relative** paths, so the coordinator could only be constructed from the
  project root. Paths now resolve through :mod:`project_paths`.
* ``DemandAgent`` was handed the whole limits mapping instead of its own
  ``demand:`` block (see :mod:`agents.demand_agent`).
* ``ACTION_NAMES[proposed_action]`` raised ``KeyError`` for any action id
  outside ``0..3`` - including a malformed RL policy output, which would take
  down the control loop and the API with it. Lookup is now total.
* The 9-feature observation was assembled inline with a function-local
  ``import numpy``. That duplicated :meth:`coordinator.rl_env.EnergyEnv._get_obs`
  and created a train/serve skew risk; both now call
  :func:`coordinator.rl_env.observation_from_state`.
* The rate limiter inside the safety layer kept relay state across evaluation
  runs. :meth:`reset` now clears it so evaluations are deterministic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import project_paths
from agents import base as agents_base
from agents.battery_agent import BatteryAgent
from agents.demand_agent import DemandAgent
from agents.price_agent import PriceAgent
from agents.solar_agent import SolarAgent
from coordinator.baseline_rules import BaselineRuleCoordinator
from coordinator.rl_env import observation_from_state
from coordinator.safety import SafetyLayer
from coordinator.train_rl import QLearningEnergyPolicy

logger = logging.getLogger("maems.coordinator")

ACTION_NAMES: Dict[int, str] = {
    0: "SOLAR_FIRST (Solar to Load, Grid covers balance)",
    1: "CHARGE_SOLAR (Surplus Solar to Battery)",
    2: "DISCHARGE_BATT (Battery supplies Load)",
    3: "CHARGE_GRID (Off-Peak Grid charges Battery)",
}
ACTION_IDLE = 0
VALID_MODES = ("rl", "rule_based")


def action_name(action: Any) -> str:
    """Total lookup for an action id - unknown ids degrade instead of raising."""
    try:
        key = int(action)
    except (TypeError, ValueError):
        return "UNKNOWN_ACTION"
    return ACTION_NAMES.get(key, f"UNKNOWN_ACTION({action})")


class MultiAgentCoordinator:
    def __init__(
        self,
        limits_path: Optional[str] = None,
        model_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.limits_path = limits_path or project_paths.LIMITS_PATH
        self.model_path = model_path or project_paths.RL_POLICY_PATH
        self.limits = project_paths.load_limits(self.limits_path)

        limits = self.limits
        batt_cfg = limits.get("battery", {}) or {}
        # One dispatch period drives the battery head-room math.
        self.control_interval_h = project_paths.control_interval_hours(limits)
        self.control_interval_sec = self.control_interval_h * 3600.0
        self.default_mode = str((limits.get("coordinator", {}) or {}).get("mode", "rule_based"))

        # Instantiate agents with their own config blocks.
        self.solar_agent = SolarAgent(limits)
        self.demand_agent = DemandAgent(limits.get("demand", {}) or {})
        self.battery_agent = BatteryAgent(batt_cfg, control_interval_h=self.control_interval_h)
        self.price_agent = PriceAgent(limits)

        # Instantiate the decision engines and the interlock.
        self.safety = SafetyLayer(limits)
        self.baseline = BaselineRuleCoordinator(limits)
        self.rl_policy = QLearningEnergyPolicy()
        self.policy_loaded = self.rl_policy.load(self.model_path, verbose=verbose)
        if not self.policy_loaded and verbose:
            logger.warning(
                "No trained policy at %s - RL mode will fall back to argmax of an "
                "empty table until you run `python -m coordinator.train_rl`.",
                project_paths.resolve(self.model_path),
            )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear interlock relay state so a replay/evaluation starts clean."""
        self.safety.reset()

    def status(self) -> Dict[str, Any]:
        """Health snapshot for ``GET /api/status`` and the Grafana header panel."""
        return {
            "agents": [self.solar_agent.name, self.demand_agent.name,
                       self.battery_agent.name, self.price_agent.name],
            "decision_engines": ["rl", "rule_based"],
            "default_mode": self.default_mode,
            "policy_loaded": self.policy_loaded,
            "control_interval_sec": round(self.control_interval_sec, 1),
            "battery_envelope": {
                "soc_min": self.safety.soc_min,
                "soc_max": self.safety.soc_max,
                "temp_max_c": self.safety.temp_max_c,
                "temp_critical_c": self.safety.temp_critical_c,
            },
            "grid_envelope": {
                "voltage_band_v": [round(self.safety.v_min, 1), round(self.safety.v_max, 1)],
                "freq_band_hz": [round(self.safety.f_min, 2), round(self.safety.f_max, 2)],
                "max_import_kw": self.safety.max_import_kw,
            },
            "interlock": {
                "rate_limit_seconds": self.safety.rate_limit_seconds,
                "trip_count": self.safety.trip_count,
                "last_trip_reason": self.safety.last_trip_reason,
            },
            "policy": self.rl_policy.policy_stats(),
        }

    # ------------------------------------------------------------------
    def process_tick(
        self,
        raw_state: Dict[str, Any],
        mode: str = "rule_based",
        current_time_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """Run one full control cycle and return the auditable decision record.

        ``current_time_sec`` is the **control-step clock** (not wall-clock): the
        safety interlock uses it for relay rate limiting, so callers replaying
        historical data should pass ``step_index * control_interval_sec``.
        """
        raw_state = dict(raw_state or {})
        if mode not in VALID_MODES:
            logger.warning("Unknown dispatch mode %r, falling back to 'rule_based'.", mode)
            mode = "rule_based"

        # 1. Collect reports from all specialist agents.
        solar_report = self.solar_agent.report(raw_state)
        demand_report = self.demand_agent.report(raw_state)
        battery_report = self.battery_agent.report(raw_state)
        price_report = self.price_agent.report(raw_state)

        agent_reports = {
            "solar": solar_report,
            "demand": demand_report,
            "battery": battery_report,
            "price": price_report,
        }

        # 2. Propose an action.
        if mode == "rl":
            obs = observation_from_state(raw_state)
            proposed_action = self.rl_policy.get_action(
                obs, evaluate=True, legal_actions=self._legal_actions(raw_state)
            )
            decision_source = "RL_POLICY"
        else:
            proposed_action = self.baseline.select_action(agent_reports, raw_state)
            decision_source = "RULE_BASED"

        # 3. Safety interlock has the final say.
        final_action, safety_status = self.safety.validate_and_override(
            proposed_action, raw_state, current_time_sec
        )

        return {
            "mode": mode,
            "decision_source": decision_source,
            "proposed_action": proposed_action,
            "proposed_action_name": action_name(proposed_action),
            "final_action": final_action,
            "final_action_name": action_name(final_action),
            "safety_status": safety_status,
            "safety_override": final_action != proposed_action,
            "control_interval_sec": round(self.control_interval_sec, 1),
            "agent_reports": agent_reports,
        }

    # ------------------------------------------------------------------
    def _legal_actions(self, raw_state: Dict[str, Any]) -> Tuple[int, ...]:
        """Mask actions the hardware envelope currently forbids.

        Mirrors the safety layer's SoC / grid guards so the RL policy is not
        penalised for proposing a move the interlock would veto anyway. The
        interlock stays the authoritative backstop - this only prevents the
        policy from earning a safety trip on every single step.
        """
        soc = agents_base.as_float(raw_state.get("soc", 0.5), 0.5)
        soc = min(max(soc, 0.0), 1.0)
        grid_status = agents_base.as_int(raw_state.get("grid_status", 1), 1)
        legal = [ACTION_IDLE]
        if soc < self.safety.soc_max:
            legal.append(1)
        if soc > self.safety.soc_min:
            legal.append(2)
        if grid_status == 1 and soc < self.safety.soc_max:
            legal.append(3)
        return tuple(legal)
