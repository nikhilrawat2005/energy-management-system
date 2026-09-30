from agents.base import BaseAgent
from typing import Dict, Any

class PriceAgent(BaseAgent):
    """
    Monitors electricity tariffs (ToD schedules), grid connection status,
    and identifies economic windows for grid-charging and load-shifting.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PriceAgent", config)

    def report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        price = float(state.get("price", 7.0))
        grid_status = int(state.get("grid_status", 1))
        hour = float(state.get("hour", 12.0))
        
        # Classify price tiers
        if price <= 5.0:
            price_tier = "OFF_PEAK_CHEAP"
            cheap_window = True
        elif price >= 10.0:
            price_tier = "PEAK_EXPENSIVE"
            cheap_window = False
        else:
            price_tier = "NORMAL"
            cheap_window = False

        grid_ok = bool(grid_status == 1)

        return {
            "agent": self.name,
            "price_inr_per_kwh": round(price, 2),
            "price_tier": price_tier,
            "cheap_window": cheap_window,
            "grid_ok": grid_ok,
            "grid_voltage_v": round(float(state.get("grid_voltage", 230.0)), 1),
            "grid_freq_hz": round(float(state.get("grid_freq", 50.0)), 2)
        }
