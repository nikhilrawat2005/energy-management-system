import psycopg2
import time
import csv
from datetime import datetime, timezone, timedelta
from coordinator.coordinator import MultiAgentCoordinator

def seed_full_timeseries_history():
    print("[Seeder] Connecting to TimescaleDB...")
    conn = psycopg2.connect(
        host="localhost",
        port=5434,
        user="maems_user",
        password="maems_password",
        dbname="maems_energy"
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    # Clean previous demo points
    cur.execute("TRUNCATE TABLE sensor_readings;")
    cur.execute("TRUNCATE TABLE agent_decisions;")
    
    rows = []
    with open("data/dataset_15min.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    coord = MultiAgentCoordinator()
    
    # We will generate a rich continuous 24-hour timeline leading up to NOW
    # 24 hours * 4 steps/hour = 96 points (every 15 mins)
    total_points = 96
    sample_rows = rows[30:30 + total_points]
    
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(hours=24)
    
    soc = 0.45
    print(f"[Seeder] Seeding {total_points} historical microgrid steps across last 24h...")
    
    for idx, r in enumerate(sample_rows):
        point_time = start_time + timedelta(minutes=idx * 15)
        
        solar = float(r["solar_kw"])
        load = float(r["load_kw"])
        price = float(r["price"])
        r["soc"] = soc
        
        res = coord.process_tick(r, mode="rule_based", current_time_sec=idx * 900.0)
        action = res["final_action"]
        action_name = res["final_action_name"]
        safety_status = res["safety_status"]
        
        # Insert sensor readings
        cur.execute(
            "INSERT INTO sensor_readings (ts, device, metric, value, unit) VALUES (%s, %s, %s, %s, %s)",
            (point_time, "solar_inverter", "solar_kw", solar, "kW")
        )
        cur.execute(
            "INSERT INTO sensor_readings (ts, device, metric, value, unit) VALUES (%s, %s, %s, %s, %s)",
            (point_time, "energy_meter", "load_kw", load, "kW")
        )
        cur.execute(
            "INSERT INTO sensor_readings (ts, device, metric, value, unit) VALUES (%s, %s, %s, %s, %s)",
            (point_time, "bms", "soc", soc, "%")
        )
        cur.execute(
            "INSERT INTO sensor_readings (ts, device, metric, value, unit) VALUES (%s, %s, %s, %s, %s)",
            (point_time, "grid_smart_meter", "price", price, "INR")
        )
        
        # Insert decisions
        cur.execute(
            "INSERT INTO agent_decisions (ts, mode, action, action_name, solar_kw, load_kw, soc, price, safety_status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (point_time, "rule_based", action, action_name, solar, load, soc, price, safety_status)
        )
        
        # Update SoC state
        if action == 1:
            soc = min(0.95, soc + 0.05)
        elif action == 2:
            soc = max(0.10, soc - 0.04)

    cur.close()
    conn.close()
    print("[Seeder] 24-hour continuous timeline populated successfully!")

if __name__ == "__main__":
    seed_full_timeseries_history()
