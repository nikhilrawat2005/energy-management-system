"""
Gym-style environment for microgrid energy management, used to train and to
score the tabular Q-learning policy.

Formulation
-----------
* **Observation (9 features, fixed contract)**
  ``[solar_kw, load_kw, soc, batt_temp, price, solar_fc_1h, load_fc_1h, sin_hour, cos_hour]``
* **Actions (4 discrete)**
  ``0 SOLAR_FIRST`` (solar serves load, grid covers the balance, battery idle)
  ``1 CHARGE_SOLAR`` (battery absorbs the solar surplus)
  ``2 DISCHARGE_BATT`` (battery serves load)
  ``3 CHARGE_GRID`` (battery absorbs off-peak grid energy)
* **Reward**
  ``-grid_cost - curtailment_penalty - battery_wear + solar_bonus - blackout_unmet``

The observation builder is exported as :func:`build_observation` and is used by
*both* the environment and :mod:`coordinator.coordinator`. Previously the
coordinator assembled the vector inline, which meant any change to one copy
silently created train/serve skew between the trained policy and production.

Bugs fixed
----------
1. **Off-by-one termination.** ``done = self.i >= self.n_steps - 1`` stopped the
   episode one step early, so the final row of the dataset was never
   simulated or learned from.
2. **Blackout credit while charging.** ``blackout_unmet`` used
   ``abs(batt_kw)``, so a battery *charging* from the grid during an outage was
   credited with the power it was drawing instead of the power it supplied.
3. **Hard-coded SoC window.** ``0.10``/``0.95`` were baked into four places
   instead of being read from ``config/limits.yaml``, so re-tuning the battery
   envelope in YAML did nothing.
4. **No empty-dataset guard.** A missing/empty CSV raised ``IndexError`` deep
   inside ``reset()`` with no actionable message.
5. **Solar under an outage was silently discarded.** During a blackout the
   solar bus is the only available source; the old flow short-circuited it.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import project_paths

# Fixed observation contract - do not reorder, the saved Q-table depends on it.
OBS_DIM = 9
N_ACTIONS = 4


def build_observation(
    solar_kw: float,
    load_kw: float,
    soc: float,
    batt_temp: float,
    price: float,
    solar_fc_1h: float,
    load_fc_1h: float,
    hour: float,
) -> np.ndarray:
    """Build the canonical 9-feature observation vector."""
    return np.array(
        [
            float(solar_kw),
            float(load_kw),
            float(soc),
            float(batt_temp),
            float(price),
            float(solar_fc_1h),
            float(load_fc_1h),
            math.sin(2.0 * math.pi * float(hour) / 24.0),
            math.cos(2.0 * math.pi * float(hour) / 24.0),
        ],
        dtype=np.float32,
    )


def observation_from_state(state: Dict[str, Any], soc: Optional[float] = None) -> np.ndarray:
    """Build the observation vector from a flat telemetry dictionary."""
    return build_observation(
        solar_kw=state.get("solar_kw", 0.0),
        load_kw=state.get("load_kw", 0.0),
        soc=state.get("soc", 0.5) if soc is None else soc,
        batt_temp=state.get("batt_temp", 27.5),
        price=state.get("price", 7.0),
        solar_fc_1h=state.get("solar_fc_1h", state.get("solar_kw", 0.0)),
        load_fc_1h=state.get("load_fc_1h", state.get("load_kw", 0.0)),
        hour=state.get("hour", 12.0),
    )


def resolve_power_flow(
    action: int,
    solar_kw: float,
    load_kw: float,
    soc: float,
    grid_status: int,
    pmax_kw: float,
    cap_kwh: float,
    eff: float,
    dt: float,
    soc_min: float,
    soc_max: float,
) -> Dict[str, Any]:
    """Microgrid power balance for a single control step - the single source of truth.

    This function is deliberately pure (no state, no I/O) so that *all three*
    consumers share one physics model:

    * :class:`EnergyEnv` when training / benchmarking the RL policy,
    * :func:`evaluate.run_strategy` for the published 7-day benchmark,
    * :mod:`ingestion.live_streamer` / :mod:`ingestion.seed_dashboard` for the
      values published to Grafana.

    Before this existed the live streamer carried its own copy of the balance
    equations, and that copy had a real bug: on ``CHARGE_GRID`` it computed
    ``grid_import = load + batt_kw``, completely ignoring on-site solar
    self-consumption, so the Grafana "Grid Import" panel overstated grid draw
    by the whole PV output on every grid-charging step.

    Returns energies in kWh and powers in kW for one ``dt``-hour step.
    """
    solar = max(0.0, float(solar_kw))
    load = max(0.0, float(load_kw))
    grid_up = 1 if int(grid_status) else 0

    act = int(action)
    if act not in (0, 1, 2, 3):
        act = 0

    soc = min(max(float(soc), float(soc_min)), float(soc_max))
    # Power that the remaining SoC envelope can absorb / deliver this step.
    charge_headroom_kw = max(0.0, (float(soc_max) - soc) * cap_kwh / dt)
    discharge_headroom_kw = max(0.0, (soc - float(soc_min)) * cap_kwh / dt)

    batt_kw = 0.0  # positive = charging, negative = discharging
    if act == 1:  # CHARGE_SOLAR - only the PV surplus may reach the battery
        batt_kw = min(pmax_kw, max(0.0, solar - load), charge_headroom_kw)
    elif act == 2:  # DISCHARGE_BATT - only the residual load may be served
        batt_kw = -min(pmax_kw, max(0.0, load - solar), discharge_headroom_kw)
    elif act == 3 and grid_up:  # CHARGE_GRID
        batt_kw = min(pmax_kw, charge_headroom_kw)

    # Round-trip efficiency: applied on the way in, inverted on the way out.
    if batt_kw > 0:
        soc += (batt_kw * eff * dt) / cap_kwh
    elif batt_kw < 0:
        soc += (batt_kw / eff * dt) / cap_kwh
    soc = float(np.clip(soc, float(soc_min), float(soc_max)))

    solar_to_load_kw = min(solar, load)
    batt_discharge_kw = max(0.0, -batt_kw)
    batt_charge_kw = max(0.0, batt_kw)

    grid_import_kw = 0.0
    if grid_up:
        # Load not covered by PV or battery comes from the utility...
        grid_import_kw = max(0.0, load - (solar_to_load_kw + batt_discharge_kw))
        # ...plus whatever the battery soaks up when grid-charging.
        if act == 3 and batt_charge_kw > 0:
            grid_import_kw += batt_charge_kw

    if batt_charge_kw > 0:
        # A charging battery is fed by PV surplus first; anything else is grid.
        solar_used_kw = solar_to_load_kw + min(
            batt_charge_kw, max(0.0, solar - solar_to_load_kw)
        )
    else:
        solar_used_kw = solar_to_load_kw
    curtailed_solar_kw = max(0.0, solar - solar_used_kw)

    # Only a downed bus can fail to serve load; a healthy grid import is not
    # "unmet" demand.
    supplied_kw = solar_to_load_kw + batt_discharge_kw
    unmet_load_kw = max(0.0, load - supplied_kw) if grid_up == 0 else 0.0

    return {
        "action": act,
        "soc": soc,
        "batt_kw": batt_kw,
        "batt_charge_kwh": batt_charge_kw * dt,
        "batt_discharge_kwh": batt_discharge_kw * dt,
        "solar_to_load_kwh": solar_to_load_kw * dt,
        "solar_used_kwh": solar_used_kw * dt,
        "curtailed_solar_kwh": curtailed_solar_kw * dt,
        "grid_import_kwh": grid_import_kw * dt,
        "grid_import_kw": grid_import_kw,
        "unmet_load_kwh": unmet_load_kw * dt,
        "grid_status": grid_up,
    }


class EnergyEnv:
    def __init__(
        self,
        data_path: str = project_paths.DEFAULT_DATASET_PATH,
        limits: Optional[Dict[str, Any]] = None,
        cap_kwh: Optional[float] = None,
        pmax_kw: Optional[float] = None,
        eff: Optional[float] = None,
        dt: Optional[float] = None,
    ):
        limits = limits if limits is not None else project_paths.load_limits()
        batt_cfg = limits.get("battery", {}) or {}
        grid_cfg = limits.get("grid", {}) or {}
        safety_cfg = limits.get("safety", {}) or {}

        self.cap_kwh = float(cap_kwh if cap_kwh is not None else batt_cfg.get("capacity_kwh", 15.0))
        self.pmax_kw = float(pmax_kw if pmax_kw is not None else grid_cfg.get("pmax_kw", 5.0))
        self.eff = float(eff if eff is not None else batt_cfg.get("efficiency", 0.95))
        # The dispatch period is the integration step for *everything*: SoC,
        # energy, cost and battery wear. Deriving it from config keeps the
        # environment, the live streamer and the evaluator on one time base.
        self.dt = float(
            dt
            if dt is not None
            else float(safety_cfg.get("rate_limit_seconds", 900)) / 3600.0
        )

        self.soc_min = float(batt_cfg.get("soc_min", 0.10))
        self.soc_max = float(batt_cfg.get("soc_max", 0.95))
        self.soc_init = float(batt_cfg.get("soc_init", 0.50))
        self.blackout_penalty = float(safety_cfg.get("blackout_penalty", 50.0))
        self.wear_cost = float(safety_cfg.get("battery_wear_cost_per_kwh", 0.02))
        self.curtailment_penalty = float(safety_cfg.get("curtailment_penalty_per_kwh", 0.50))
        self.solar_bonus = float(safety_cfg.get("solar_bonus_per_kwh", 0.10))

        self.data_path = project_paths.resolve(data_path)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Training dataset not found: {self.data_path}")

        self.rows: List[Dict[str, str]] = []
        with open(self.data_path, "r", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Training dataset is empty: {self.data_path}")

        required = {"hour", "solar_kw", "load_kw", "price", "grid_status", "batt_temp"}
        missing = required.difference(self.rows[0].keys())
        if missing:
            raise ValueError(
                f"Training dataset {self.data_path} is missing column(s): "
                f"{', '.join(sorted(missing))}"
            )

        self.n_steps = len(self.rows)
        self.action_space_n = N_ACTIONS
        self.obs_dim = OBS_DIM
        self.i = 0
        self.soc = self.soc_init

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.i = 0
        self.soc = self.soc_init
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        r = self.rows[self.i]
        return build_observation(
            solar_kw=r["solar_kw"],
            load_kw=r["load_kw"],
            soc=self.soc,
            batt_temp=r["batt_temp"],
            price=r["price"],
            solar_fc_1h=r.get("solar_fc_1h", r["solar_kw"]),
            load_fc_1h=r.get("load_fc_1h", r["load_kw"]),
            hour=r["hour"],
        )

    # ------------------------------------------------------------------
    def legal_actions(self, soc: Optional[float] = None, grid_status: Optional[int] = None) -> List[int]:
        """Actions that the hardware envelope currently permits.

        Used by the policy to mask illegal moves, so the learned policy never
        *proposes* a charge at the SoC ceiling or a grid charge during a
        blackout. The safety interlock remains the independent backstop; masking
        just stops the policy from tripping it on every single step.
        """
        soc = self.soc if soc is None else soc
        r = self.rows[min(self.i, self.n_steps - 1)]
        gs = int(r["grid_status"]) if grid_status is None else int(grid_status)
        legal = [0]
        if soc < self.soc_max:
            legal.append(1)
        if soc > self.soc_min:
            legal.append(2)
        if gs == 1 and soc < self.soc_max:
            legal.append(3)
        return legal

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        r = self.rows[self.i]
        solar = float(r["solar_kw"])
        load = float(r["load_kw"])
        price = float(r["price"])
        grid_status = int(r["grid_status"])

        flow = resolve_power_flow(
            action=action,
            solar_kw=solar,
            load_kw=load,
            soc=self.soc,
            grid_status=grid_status,
            pmax_kw=self.pmax_kw,
            cap_kwh=self.cap_kwh,
            eff=self.eff,
            dt=self.dt,
            soc_min=self.soc_min,
            soc_max=self.soc_max,
        )
        self.soc = flow["soc"]
        batt_kw = flow["batt_kw"]
        grid_cost = flow["grid_import_kwh"] * price
        battery_wear = self.wear_cost * abs(batt_kw) * self.dt
        solar_bonus = self.solar_bonus * flow["solar_used_kwh"]
        curtailment_penalty = self.curtailment_penalty * flow["curtailed_solar_kwh"]
        # `unmet_load_kwh` is only non-zero while the bus is down; during normal
        # operation every imported kWh was legitimately served by the utility and
        # must not be counted as unmet demand.
        blackout_unmet = self.blackout_penalty * flow["unmet_load_kwh"]

        reward = (
            (flow["batt_discharge_kwh"] - flow["batt_charge_kwh"]) * price
            - curtailment_penalty
            - battery_wear
            + solar_bonus
            - blackout_unmet
        )

        self.i += 1
        # Off-by-one fix: the episode now ends only after the *last* row has
        # been simulated.
        done = self.i >= self.n_steps

        info = {
            "step": self.i,
            "action": flow["action"],
            "solar_kw": solar,
            "load_kw": load,
            "batt_kw": batt_kw,
            "soc": self.soc,
            "grid_import_kwh": flow["grid_import_kwh"],
            "grid_cost_inr": grid_cost,
            "solar_used_kwh": flow["solar_used_kwh"],
            "curtailed_solar_kwh": flow["curtailed_solar_kwh"],
            "unmet_load_kwh": flow["unmet_load_kwh"],
            "price": price,
            "grid_status": grid_status,
        }

        next_obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return next_obs, float(reward), done, info
