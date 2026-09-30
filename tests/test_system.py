import unittest
from coordinator.coordinator import MultiAgentCoordinator
from coordinator.safety import SafetyLayer
from forecasting.features import SimpleGradientBoostingRegressor
import numpy as np

class TestMAEMSSystem(unittest.TestCase):
    def setUp(self):
        self.coord = MultiAgentCoordinator()
        self.safety = self.coord.safety

    def test_battery_thermal_critical_lockout(self):
        # Temp >= 55C must lockout battery flow
        state = {"soc": 0.5, "batt_temp": 56.0, "grid_status": 1}
        action, msg = self.safety.validate_and_override(1, state, 100.0)
        self.assertEqual(action, 0)
        self.assertIn("Critical limit", msg)

    def test_battery_overcharge_prevention(self):
        # SoC >= 95% cannot charge further
        state = {"soc": 0.96, "batt_temp": 30.0, "grid_status": 1}
        action, msg = self.safety.validate_and_override(1, state, 100.0)
        self.assertEqual(action, 0)
        self.assertIn("OVERRIDE", msg)

    def test_battery_overdischarge_prevention(self):
        # SoC <= 10% cannot discharge
        state = {"soc": 0.08, "batt_temp": 30.0, "grid_status": 1}
        action, msg = self.safety.validate_and_override(2, state, 100.0)
        self.assertEqual(action, 0)
        self.assertIn("OVERRIDE", msg)

    def test_grid_blackout_island_mode(self):
        # Grid failure forces islanding & blocks grid charge
        state = {"soc": 0.60, "batt_temp": 30.0, "grid_status": 0}
        action, msg = self.safety.validate_and_override(3, state, 100.0)
        self.assertEqual(action, 0)
        self.assertIn("Grid Blackout", msg)

    def test_solar_agent_surplus(self):
        rep = self.coord.solar_agent.report({"solar_kw": 8.0, "load_kw": 3.0, "irradiance_wm2": 800, "hour": 13.0})
        self.assertGreater(rep["current_surplus_kw"], 4.5)
        self.assertEqual(rep["status"], "peak_production")

    def test_demand_agent_peak_risk(self):
        rep = self.coord.demand_agent.report({"load_kw": 7.5, "load_fc_1h": 8.0})
        self.assertTrue(rep["peak_risk"])
        self.assertEqual(rep["load_category"], "critical_high")

    def test_price_agent_peak_tier(self):
        rep = self.coord.price_agent.report({"price": 11.5, "grid_status": 1})
        self.assertEqual(rep["price_tier"], "PEAK_EXPENSIVE")
        self.assertFalse(rep["cheap_window"])

    def test_price_agent_offpeak_tier(self):
        rep = self.coord.price_agent.report({"price": 4.5, "grid_status": 1})
        self.assertEqual(rep["price_tier"], "OFF_PEAK_CHEAP")
        self.assertTrue(rep["cheap_window"])

    def test_ml_demand_model_inference(self):
        model = SimpleGradientBoostingRegressor()
        model.load("models/demand_forecast.json")
        sample = np.array([[12.0, 2.0, 0.0, 28.0, 50.0, 3.5, 3.2]], dtype=np.float32)
        pred = model.predict(sample)[0]
        self.assertGreater(pred, 0.5)

    def test_ml_solar_model_inference(self):
        model = SimpleGradientBoostingRegressor()
        model.load("models/solar_forecast.json")
        sample = np.array([[13.0, 800.0, 32.0, 6.0, 5.8]], dtype=np.float32)
        pred = model.predict(sample)[0]
        self.assertGreater(pred, 3.0)

if __name__ == "__main__":
    unittest.main()
