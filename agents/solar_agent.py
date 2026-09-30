from agents.base import BaseAgent
from typing import Dict, Any

class SolarAgent(BaseAgent):
    """
    Monitors solar generation, irradiance, and horizon forecasts.
    Evaluates current and near-term expected solar surplus.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SolarAgent", config)

    def report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        solar_now = float(state.get("solar_kw", 0.0))
        solar_fc_1h = float(state.get("solar_fc_1h", 0.0))
        solar_fc_4h = float(state.get("solar_fc_4h", 0.0))
        load_now = float(state.get("load_kw", 0.0))
        irradiance = float(state.get("irradiance_wm2", 0.0))
        hour = float(state.get("hour", 0.0))
        
        current_surplus = max(0.0, solar_now - load_now)
        expected_surplus_1h = max(0.0, solar_fc_1h - float(state.get("load_fc_1h", load_now)))
        
        status = "normal"
        if solar_now > 7.0:
            status = "peak_production"
        elif solar_now < 0.1 and (6.0 <= hour <= 18.0):
            status = "cloud_drop_alarm"

        return {
            "agent": self.name,
            "solar_now_kw": round(solar_now, 3),
            "solar_fc_1h_kw": round(solar_fc_1h, 3),
            "solar_fc_4h_kw": round(solar_fc_4h, 3),
            "irradiance_wm2": round(irradiance, 1),
            "current_surplus_kw": round(current_surplus, 3),
            "expected_surplus_1h_kw": round(expected_surplus_1h, 3),
            "status": status
        }
