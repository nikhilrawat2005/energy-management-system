"""Quick test - Ollama AI reasoning end-to-end."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.ai_coordinator_agent import generate_ai_reasoning

test_state = {
    "solar_kw": 4.2, "load_kw": 3.1, "soc": 0.68, "batt_temp": 29.0,
    "price": 8.5, "grid_status": 1, "grid_voltage": 230.0, "grid_freq": 50.0,
    "hour": 14.0, "irradiance_wm2": 780.0,
    "solar_fc_1h": 3.8, "load_fc_1h": 3.4,
    "solar_fc_4h": 1.2, "load_fc_4h": 4.5,
}
test_solar_rep = {"current_surplus_kw": 1.1, "health_flag": "HEALTHY"}
test_demand_rep = {"peak_risk": False}
test_battery_rep = {"health_flag": "HEALTHY", "avail_charge_kw": 2.0, "avail_discharge_kw": 3.0}
test_price_rep  = {"tariff_tier": "PEAK_EXPENSIVE"}

print("Testing AI Reasoning Agent...")
result = generate_ai_reasoning(
    state=test_state,
    solar_rep=test_solar_rep,
    demand_rep=test_demand_rep,
    battery_rep=test_battery_rep,
    price_rep=test_price_rep,
    action_name="CHARGE_SOLAR",
    safety_status="SAFE_CONFIRMED"
)
print(f"\nSource  : {result['source']}")
print(f"Model   : {result['model']}")
print(f"Latency : {result['latency_ms']} ms")
print(f"\nReasoning:\n{result['reasoning']}")
