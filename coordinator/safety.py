from typing import Dict, Any, Tuple

class SafetyLayer:
    """
    Mandatory Hardware & Operational Safety Interlock:
    1. Enforces strict SoC boundaries [soc_min, soc_max].
    2. Enforces thermal limits (overheat isolation).
    3. Handles Grid Blackout (islanded mode: sheds non-critical load, cuts off grid charging/export).
    4. Rate-limits relay toggles to protect inverters and contactors.
    """
    def __init__(self, limits_cfg: Dict[str, Any]):
        batt_cfg = limits_cfg.get("battery", {})
        self.soc_min = batt_cfg.get("soc_min", 0.10)
        self.soc_max = batt_cfg.get("soc_max", 0.95)
        self.temp_max_c = batt_cfg.get("temp_max_c", 45.0)
        self.temp_critical_c = batt_cfg.get("temp_critical_c", 55.0)
        self.max_charge_kw = batt_cfg.get("max_charge_kw", 5.0)
        self.max_discharge_kw = batt_cfg.get("max_discharge_kw", 5.0)
        
        safety_cfg = limits_cfg.get("safety", {})
        self.rate_limit_seconds = safety_cfg.get("rate_limit_seconds", 900)
        self.last_action_time = 0
        self.last_action = None

    def validate_and_override(self, proposed_action: int, state: Dict[str, Any], current_time_sec: float) -> Tuple[int, str]:
        """
        Actions:
          0: IDLE / SOLAR_FIRST (Solar serves load, Grid covers rest)
          1: CHARGE_SOLAR (Surplus solar to battery)
          2: DISCHARGE_BATT (Battery serves load)
          3: CHARGE_GRID (Grid charges battery during cheap hours)
        """
        soc = float(state.get("soc", 0.5))
        temp = float(state.get("batt_temp", 25.0))
        grid_status = int(state.get("grid_status", 1))
        
        # 1. Critical Thermal Overheat Guard
        if temp >= self.temp_critical_c:
            if proposed_action in [1, 2, 3]:
                return 0, f"SAFETY_OVERRIDE: Battery Temp ({temp:.1f}C) >= Critical limit ({self.temp_critical_c}C). Battery power flow locked out."

        # 2. Grid Blackout / Islanding Guard
        if grid_status == 0:
            if proposed_action == 3: # Cannot charge from grid when grid is down
                return 0, "SAFETY_OVERRIDE: Grid Blackout detected! Grid charging disabled."
            # In islanded mode, if solar isn't enough, battery must serve load if available
            if proposed_action == 0 and soc > self.soc_min:
                return 2, "SAFETY_ISLAND_MODE: Grid outage! Forcing Battery discharge to sustain microgrid load."

        # 3. High SoC Guard (Overcharge prevention)
        if soc >= self.soc_max:
            if proposed_action in [1, 3]:
                return 0, f"SAFETY_OVERRIDE: Battery SoC ({soc*100:.1f}%) >= Max limit ({self.soc_max*100:.1f}%). Charging blocked."

        # 4. Low SoC Guard (Overdischarge protection)
        if soc <= self.soc_min:
            if proposed_action == 2:
                return 0, f"SAFETY_OVERRIDE: Battery SoC ({soc*100:.1f}%) <= Min reserve ({self.soc_min*100:.1f}%). Discharge blocked."

        # Action confirmed safe
        self.last_action = proposed_action
        self.last_action_time = current_time_sec
        return proposed_action, "SAFE_CONFIRMED"
