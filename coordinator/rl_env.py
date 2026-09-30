import numpy as np
import csv
from typing import Dict, Any, Tuple

class EnergyEnv:
    """
    Gym-compatible Environment for Microgrid Energy Management.
    Formulation:
      - State (9 features): [solar_kw, load_kw, soc, batt_temp, price, solar_fc_1h, load_fc_1h, sin_hour, cos_hour]
      - Actions (4 discrete):
          0: SOLAR_FIRST (solar serves load, grid covers balance, batt idle)
          1: CHARGE_SOLAR (charge battery from solar surplus)
          2: DISCHARGE_BATT (discharge battery to meet load)
          3: CHARGE_GRID (charge battery from cheap grid)
      - Reward:
          -(grid_cost) - 0.5 * curtailed_solar - battery_wear + 0.1 * solar_used
    """
    def __init__(self, data_path: str = "data/dataset_15min.csv", cap_kwh: float = 15.0, pmax_kw: float = 5.0, eff: float = 0.95, dt: float = 0.25):
        self.cap_kwh = cap_kwh
        self.pmax_kw = pmax_kw
        self.eff = eff
        self.dt = dt # 15 minutes = 0.25h
        
        # Load dataset
        self.rows = []
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)
                
        self.n_steps = len(self.rows)
        self.action_space_n = 4
        self.obs_dim = 9
        self.i = 0
        self.soc = 0.5

    def reset(self) -> np.ndarray:
        self.i = 0
        self.soc = 0.5
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        r = self.rows[self.i]
        h = float(r["hour"])
        return np.array([
            float(r["solar_kw"]),
            float(r["load_kw"]),
            float(self.soc),
            float(r["batt_temp"]),
            float(r["price"]),
            float(r["solar_fc_1h"]),
            float(r["load_fc_1h"]),
            np.sin(2.0 * np.pi * h / 24.0),
            np.cos(2.0 * np.pi * h / 24.0)
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        r = self.rows[self.i]
        solar = float(r["solar_kw"])
        load = float(r["load_kw"])
        price = float(r["price"])
        grid_status = int(r["grid_status"])
        
        batt_kw = 0.0  # + charge, - discharge
        
        if action == 1: # Charge from solar surplus
            surplus = max(0.0, solar - load)
            max_can_charge = (0.95 - self.soc) * self.cap_kwh / self.dt
            batt_kw = min(self.pmax_kw, surplus, max_can_charge)
        elif action == 2: # Discharge to meet load
            deficit = max(0.0, load - solar)
            max_can_discharge = (self.soc - 0.10) * self.cap_kwh / self.dt
            batt_kw = -min(self.pmax_kw, deficit, max_can_discharge)
        elif action == 3 and grid_status == 1: # Charge from grid
            max_can_charge = (0.95 - self.soc) * self.cap_kwh / self.dt
            batt_kw = min(self.pmax_kw, max_can_charge)

        # Update battery SoC
        if batt_kw > 0:
            self.soc += (batt_kw * self.eff * self.dt) / self.cap_kwh
        elif batt_kw < 0:
            self.soc += (batt_kw / self.eff * self.dt) / self.cap_kwh
            
        self.soc = float(np.clip(self.soc, 0.10, 0.95))

        # Power balance
        solar_to_load = min(solar, load)
        curtailed_solar = 0.0
        
        if batt_kw > 0: # charging
            # solar first covers load, then battery
            solar_used = solar_to_load + batt_kw
            curtailed_solar = max(0.0, solar - solar_used)
            shortfall = max(0.0, load - solar_to_load)
            grid_import = shortfall if grid_status == 1 else 0.0
            if action == 3: # charging from grid
                grid_import += batt_kw
        else: # discharging or idle
            batt_discharge = abs(batt_kw)
            supplied = solar_to_load + batt_discharge
            shortfall = max(0.0, load - supplied)
            grid_import = shortfall if grid_status == 1 else 0.0
            curtailed_solar = max(0.0, solar - solar_to_load)
            solar_used = solar_to_load

        # Cost and reward calculation
        grid_cost = grid_import * price * self.dt
        batt_wear = 0.02 * abs(batt_kw) * self.dt
        solar_bonus = 0.10 * solar_used * self.dt
        curtailment_penalty = 0.50 * curtailed_solar * self.dt
        blackout_unmet = 50.0 * max(0.0, load - (solar_to_load + abs(batt_kw))) if grid_status == 0 else 0.0

        reward = -(grid_cost) - curtailment_penalty - batt_wear + solar_bonus - blackout_unmet

        self.i += 1
        done = self.i >= self.n_steps - 1
        
        info = {
            "step": self.i,
            "action": action,
            "solar_kw": solar,
            "load_kw": load,
            "batt_kw": batt_kw,
            "soc": self.soc,
            "grid_import_kwh": grid_import * self.dt,
            "grid_cost_inr": grid_cost,
            "solar_used_kwh": solar_used * self.dt,
            "curtailed_solar_kwh": curtailed_solar * self.dt,
            "price": price,
            "grid_status": grid_status
        }
        
        next_obs = self._get_obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return next_obs, float(reward), done, info
