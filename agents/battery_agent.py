from agents.base import BaseAgent
from typing import Dict, Any

class BatteryAgent(BaseAgent):
    """
    Monitors Battery State of Charge (SoC), thermal health,
    and dynamically enforces charge/discharge capability limits.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BatteryAgent", config)
        self.capacity_kwh = config.get("capacity_kwh", 15.0)
        self.soc_min = config.get("soc_min", 0.10)
        self.soc_max = config.get("soc_max", 0.95)
        self.max_charge_kw = config.get("max_charge_kw", 5.0)
        self.max_discharge_kw = config.get("max_discharge_kw", 5.0)
        self.temp_max_c = config.get("temp_max_c", 45.0)
        self.temp_critical_c = config.get("temp_critical_c", 55.0)

    def report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        soc = float(state.get("soc", 0.5))
        temp = float(state.get("batt_temp", 28.0))
        
        # Thermal health checks
        is_critical_hot = temp >= self.temp_critical_c
        is_throttled = temp >= self.temp_max_c
        
        # Determine available power flow
        if is_critical_hot:
            avail_charge = 0.0
            avail_discharge = 0.0
            health_flag = "CRITICAL_OVERHEAT_LOCKOUT"
        elif is_throttled:
            avail_charge = min(self.max_charge_kw * 0.5, (self.soc_max - soc) * self.capacity_kwh)
            avail_discharge = min(self.max_discharge_kw * 0.5, (soc - self.soc_min) * self.capacity_kwh)
            health_flag = "THERMAL_THROTTLING"
        else:
            avail_charge = 0.0 if soc >= self.soc_max else self.max_charge_kw
            avail_discharge = 0.0 if soc <= self.soc_min else self.max_discharge_kw
            health_flag = "HEALTHY"

        return {
            "agent": self.name,
            "soc": round(soc, 4),
            "soc_pct": round(soc * 100.0, 1),
            "batt_temp_c": round(temp, 2),
            "avail_charge_kw": round(max(0.0, avail_charge), 3),
            "avail_discharge_kw": round(max(0.0, avail_discharge), 3),
            "stored_energy_kwh": round(soc * self.capacity_kwh, 2),
            "health_flag": health_flag
        }
