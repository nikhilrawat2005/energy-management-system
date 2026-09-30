from agents.base import BaseAgent
from typing import Dict, Any

class DemandAgent(BaseAgent):
    """
    Monitors consumption load, detects spikes, and evaluates peak demand risk.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DemandAgent", config)
        self.peak_threshold_kw = config.get("peak_threshold_kw", 6.0)

    def report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        load_now = float(state.get("load_kw", 0.0))
        load_fc_1h = float(state.get("load_fc_1h", 0.0))
        load_fc_4h = float(state.get("load_fc_4h", 0.0))
        
        peak_risk = (load_now >= self.peak_threshold_kw) or (load_fc_1h >= self.peak_threshold_kw)
        
        load_category = "low"
        if load_now > self.peak_threshold_kw:
            load_category = "critical_high"
        elif load_now > 3.0:
            load_category = "moderate"

        return {
            "agent": self.name,
            "load_now_kw": round(load_now, 3),
            "load_fc_1h_kw": round(load_fc_1h, 3),
            "load_fc_4h_kw": round(load_fc_4h, 3),
            "load_category": load_category,
            "peak_risk": bool(peak_risk)
        }
