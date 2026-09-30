import unittest
from agents.ai_coordinator_agent import generate_ai_reasoning, _fallback_reasoning

class TestAIAgent(unittest.TestCase):
    def setUp(self):
        self.state = {
            "solar_kw": 4.5,
            "load_kw": 2.1,
            "soc": 0.65,
            "batt_temp": 28.0,
            "price": 7.5,
            "grid_status": 1,
            "hour": 13.0,
            "solar_fc_1h": 4.0,
            "load_fc_1h": 2.5,
            "solar_fc_4h": 1.5,
            "load_fc_4h": 3.0,
        }
        self.solar_rep = {"current_surplus_kw": 2.4, "health_flag": "HEALTHY"}
        self.demand_rep = {"peak_risk": False}
        self.battery_rep = {"health_flag": "HEALTHY", "avail_charge_kw": 3.0, "avail_discharge_kw": 3.0}
        self.price_rep = {"tariff_tier": "NORMAL"}

    def test_ai_fallback_reasoning_generation(self):
        """Verify deterministic fallback reasoning works immediately without external service."""
        text = _fallback_reasoning(
            self.state, self.solar_rep, self.demand_rep, self.battery_rep, self.price_rep,
            "CHARGE_SOLAR", "SAFE_CONFIRMED"
        )
        self.assertIn("Solar", text)
        self.assertIn("Load", text)
        self.assertIn("RECOMMEND:", text)

    def test_ai_coordinator_entrypoint_schema(self):
        """Verify generate_ai_reasoning returns expected keys and contract."""
        res = generate_ai_reasoning(
            self.state, self.solar_rep, self.demand_rep, self.battery_rep, self.price_rep,
            "CHARGE_SOLAR", "SAFE_CONFIRMED"
        )
        self.assertIn("reasoning", res)
        self.assertIn("source", res)
        self.assertIn("model", res)
        self.assertIn("latency_ms", res)
        self.assertIn(res["source"], ["ollama", "fallback"])

if __name__ == "__main__":
    unittest.main()
