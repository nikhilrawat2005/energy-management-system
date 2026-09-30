import time
import psycopg2
import csv
from datetime import datetime, timezone, timedelta
from coordinator.coordinator import MultiAgentCoordinator
from forecasting.features import SimpleGradientBoostingRegressor
import numpy as np

def run_realtime_streamer(dataset_path="data/dataset_15min.csv", interval_sec=2.0):
    """
    Live Microgrid Streamer:
    - Runs in continuous loop every 2 seconds (simulating 15-min microgrid tick)
    - Computes real-time Demand & Solar Predictions using our trained ML models
    - Computes Cost Savings vs Grid-Only baseline
    - Executes Multi-Agent Coordinator (Solar, Demand, Battery, Price agents)
    - Records Live Animated Data into TimescaleDB for moving Grafana dashboards
    """
    print("[Live Streamer] Connecting to TimescaleDB (port 5434)...")
    conn = psycopg2.connect(
        host="localhost",
        port=5434,
        user="maems_user",
        password="maems_password",
        dbname="maems_energy"
    )
    conn.autocommit = True
    cur = conn.cursor()
    print("[Live Streamer] Connected successfully!")

    coord = MultiAgentCoordinator()
    
    # Load trained ML models for live predictions
    demand_model = SimpleGradientBoostingRegressor()
    solar_model = SimpleGradientBoostingRegressor()
    demand_model.load("models/demand_forecast.json")
    solar_model.load("models/solar_forecast.json")
    print("[Live Streamer] Loaded ML Forecasters successfully!")

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    soc = 0.50
    step_idx = 0
    total_savings_inr = 0.0

    print(f"[Live Streamer] Starting live streaming tick every {interval_sec}s...")

    while True:
        r = rows[step_idx % len(rows)]
        now = datetime.now(timezone.utc)
        
        solar = float(r["solar_kw"])
        load = float(r["load_kw"])
        price = float(r["price"])
        irradiance = float(r["irradiance_wm2"])
        ambient_temp = float(r["ambient_temp"])
        hour = float(r["hour"])
        day_of_week = float(r["day_of_week"])
        is_weekend = float(r["is_weekend"])
        
        # 1. Live ML Predictions (t+1h ahead)
        demand_feats = np.array([[hour, day_of_week, is_weekend, ambient_temp, float(r["humidity"]), load, load]], dtype=np.float32)
        solar_feats = np.array([[hour, irradiance, ambient_temp, solar, solar]], dtype=np.float32)
        pred_demand_1h = max(0.4, float(demand_model.predict(demand_feats)[0]))
        pred_solar_1h = max(0.0, float(solar_model.predict(solar_feats)[0]))

        r_state = dict(r)
        r_state["soc"] = soc
        r_state["solar_fc_1h"] = pred_solar_1h
        r_state["load_fc_1h"] = pred_demand_1h

        # 2. Coordinator Execution (Multi-Agent + Safety)
        res = coord.process_tick(r_state, mode="rule_based", current_time_sec=time.time())
        final_action = res["final_action"]
        final_name = res["final_action_name"]
        safety_status = res["safety_status"]

        # 3. Microgrid Power Balance & Savings Calculation
        # Grid-only cost: load * price * 0.25h
        grid_only_cost = load * price * 0.25
        
        batt_kw = 0.0
        if final_action == 1: # charge from solar surplus
            surplus = max(0.0, solar - load)
            batt_kw = min(5.0, surplus, (0.95 - soc) * 15.0 / 0.25)
            soc = min(0.95, soc + (batt_kw * 0.95 * 0.25) / 15.0)
            actual_grid_import = max(0.0, load - min(solar, load))
        elif final_action == 2: # discharge to serve load
            deficit = max(0.0, load - solar)
            batt_kw = -min(5.0, deficit, (soc - 0.10) * 15.0 / 0.25)
            soc = max(0.10, soc - (abs(batt_kw) / 0.95 * 0.25) / 15.0)
            actual_grid_import = max(0.0, load - (min(solar, load) + abs(batt_kw)))
        elif final_action == 3: # charge from off-peak grid
            batt_kw = min(5.0, (0.95 - soc) * 15.0 / 0.25)
            soc = min(0.95, soc + (batt_kw * 0.95 * 0.25) / 15.0)
            actual_grid_import = load + batt_kw
        else:
            actual_grid_import = max(0.0, load - min(solar, load))

        maems_cost = actual_grid_import * price * 0.25
        tick_saving = max(0.0, grid_only_cost - maems_cost)
        total_savings_inr += tick_saving

        # 4. Insert Live Sensor & Forecast Telemetry
        metrics = [
            ("solar_inverter", "solar_kw", solar, "kW"),
            ("energy_meter", "load_kw", load, "kW"),
            ("bms", "soc", soc, "%"),
            ("grid_smart_meter", "price", price, "INR"),
            ("solar_forecaster_ml", "pred_solar_1h", pred_solar_1h, "kW"),
            ("demand_forecaster_ml", "pred_load_1h", pred_demand_1h, "kW"),
            ("economic_calculator", "total_savings_inr", total_savings_inr, "INR"),
            ("grid_smart_meter", "grid_import_kw", actual_grid_import, "kW")
        ]

        for dev, m_name, val, unit in metrics:
            cur.execute(
                "INSERT INTO sensor_readings (ts, device, metric, value, unit) VALUES (%s, %s, %s, %s, %s)",
                (now, dev, m_name, val, unit)
            )

        cur.execute(
            "INSERT INTO agent_decisions (ts, mode, action, action_name, solar_kw, load_kw, soc, price, safety_status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (now, "rule_based", final_action, final_name, solar, load, soc, price, safety_status)
        )

        step_idx += 1
        time.sleep(interval_sec)

if __name__ == "__main__":
    run_realtime_streamer()
