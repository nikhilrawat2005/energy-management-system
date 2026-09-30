from typing import Dict, Any

class BaselineRuleCoordinator:
    """
    Standard deterministic energy management heuristic:
    - If solar > load: Charge battery with solar surplus
    - Elif price is peak (high) and battery has charge: Discharge battery to save peak costs
    - Elif price is cheap (off-peak) and battery is low: Charge battery from cheap grid
    - Else: Solar serves load, grid balances shortfall
    """
    def __init__(self, limits_cfg: Dict[str, Any]):
        batt_cfg = limits_cfg.get("battery", {})
        self.soc_min = batt_cfg.get("soc_min", 0.10)
        self.soc_max = batt_cfg.get("soc_max", 0.95)

    def select_action(self, agent_reports: Dict[str, Dict[str, Any]], raw_state: Dict[str, Any]) -> int:
        solar_rep = agent_reports.get("solar", {})
        batt_rep = agent_reports.get("battery", {})
        price_rep = agent_reports.get("price", {})
        
        solar_now = solar_rep.get("solar_now_kw", 0.0)
        load_now = agent_reports.get("demand", {}).get("load_now_kw", 0.0)
        soc = batt_rep.get("soc", 0.5)
        price_tier = price_rep.get("price_tier", "NORMAL")
        
        # Rule 1: Surplus solar energy
        if solar_now > load_now and soc < self.soc_max:
            return 1  # CHARGE_SOLAR
            
        # Rule 2: Peak tariff shaving
        if price_tier == "PEAK_EXPENSIVE" and soc > (self.soc_min + 0.15):
            return 2  # DISCHARGE_BATT
            
        # Rule 3: Economic grid charging during off-peak windows
        if price_tier == "OFF_PEAK_CHEAP" and soc < 0.60:
            return 3  # CHARGE_GRID
            
        # Rule 4: Default solar-first, grid shortfall
        return 0  # IDLE / SOLAR_FIRST
