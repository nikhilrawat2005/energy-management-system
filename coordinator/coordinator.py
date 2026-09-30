import yaml
import os
from typing import Dict, Any, Tuple

from agents.solar_agent import SolarAgent
from agents.demand_agent import DemandAgent
from agents.battery_agent import BatteryAgent
from agents.price_agent import PriceAgent
from coordinator.safety import SafetyLayer
from coordinator.baseline_rules import BaselineRuleCoordinator
from coordinator.train_rl import QLearningEnergyPolicy

ACTION_NAMES = {
    0: "SOLAR_FIRST (Solar to Load, Grid covers balance)",
    1: "CHARGE_SOLAR (Surplus Solar to Battery)",
    2: "DISCHARGE_BATT (Battery supplies Load)",
    3: "CHARGE_GRID (Off-Peak Grid charges Battery)"
}

class MultiAgentCoordinator:
    """
    Central Coordinator managing:
    1. Polling Specialized Agents (Solar, Demand, Battery, Price).
    2. Deciding action using RL Policy or Baseline Rules.
    3. Passing proposed action through Safety Layer to prevent battery damage & handle outages.
    """
    def __init__(self, limits_path: str = "config/limits.yaml", model_path: str = "models/rl_policy.json"):
        with open(limits_path, "r", encoding="utf-8") as f:
            self.limits = yaml.safe_load(f)
            
        # Instantiate Agents
        self.solar_agent = SolarAgent(self.limits)
        self.demand_agent = DemandAgent(self.limits)
        self.battery_agent = BatteryAgent(self.limits.get("battery", {}))
        self.price_agent = PriceAgent(self.limits)
        
        # Instantiate Coordinator logic
        self.safety = SafetyLayer(self.limits)
        self.baseline = BaselineRuleCoordinator(self.limits)
        self.rl_policy = QLearningEnergyPolicy()
        if os.path.exists(model_path):
            self.rl_policy.load(model_path)
            
    def process_tick(self, raw_state: Dict[str, Any], mode: str = "rl", current_time_sec: float = 0.0) -> Dict[str, Any]:
        """
        Processes a single control step:
        - raw_state: dictionary with sensor metrics
        - mode: 'rl' or 'rule_based'
        """
        # 1. Collect reports from all specialized agents
        solar_report = self.solar_agent.report(raw_state)
        demand_report = self.demand_agent.report(raw_state)
        battery_report = self.battery_agent.report(raw_state)
        price_report = self.price_agent.report(raw_state)
        
        agent_reports = {
            "solar": solar_report,
            "demand": demand_report,
            "battery": battery_report,
            "price": price_report
        }
        
        # 2. Propose Action
        if mode == "rl":
            import numpy as np
            h = float(raw_state.get("hour", 12.0))
            obs = np.array([
                float(raw_state.get("solar_kw", 0.0)),
                float(raw_state.get("load_kw", 0.0)),
                float(raw_state.get("soc", 0.5)),
                float(raw_state.get("batt_temp", 25.0)),
                float(raw_state.get("price", 7.0)),
                float(raw_state.get("solar_fc_1h", 0.0)),
                float(raw_state.get("load_fc_1h", 0.0)),
                np.sin(2.0 * np.pi * h / 24.0),
                np.cos(2.0 * np.pi * h / 24.0)
            ], dtype=np.float32)
            proposed_action = self.rl_policy.get_action(obs, evaluate=True)
            decision_source = "RL_POLICY"
        else:
            proposed_action = self.baseline.select_action(agent_reports, raw_state)
            decision_source = "RULE_BASED"
            
        # 3. Apply Safety Interlock Layer
        final_action, safety_status = self.safety.validate_and_override(proposed_action, raw_state, current_time_sec)
        
        return {
            "mode": mode,
            "decision_source": decision_source,
            "proposed_action": proposed_action,
            "proposed_action_name": ACTION_NAMES[proposed_action],
            "final_action": final_action,
            "final_action_name": ACTION_NAMES[final_action],
            "safety_status": safety_status,
            "agent_reports": agent_reports
        }
