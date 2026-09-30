import psycopg2
import time
import csv
from datetime import datetime, timezone
from coordinator.coordinator import MultiAgentCoordinator

def stream_telemetry_to_timescale(dataset_path="data/dataset_15min.csv", host="localhost", port=5432):
    print(f"[DB Feeder] Connecting to TimescaleDB at {host}:{port}...")
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user="maems_user",
            password="maems_password",
            dbname="maems_energy"
        )
        conn.autocommit = True
        cur = conn.cursor()
        print("[DB Feeder] Connected successfully!")
    except Exception as e:
        print(f"[DB Feeder] Database connection failed: {e}")
        return

    coord = MultiAgentCoordinator()
    
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    print(f"[DB Feeder] Streaming {len(rows)} data points into TimescaleDB and executing Multi-Agent control...")
    
    soc = 0.50
    # Stream in a real-time loop (e.g. 50 recent points with current timestamps)
    for i in range(len(rows) - 60, len(rows)):
        r = rows[i]
        r["soc"] = soc
        
        now = datetime.now(timezone.utc)
        
        # Process Multi-Agent Coordinator Tick
        res = coord.process_tick(r, mode="rule_based", current_time_sec=time.time())
        final_action = res["final_action"]
        final_name = res["final_action_name"]
        safety_status = res["safety_status"]
        
        solar = float(r["solar_kw"])
        load = float(r["load_kw"])
        price = float(r["price"])
        
        # Insert sensor readings
        cur.execute(
            "INSERT INTO sensor_readings (ts, device, metric, value, unit) VALUES (%s, %s, %s, %s, %s)",
            (now, "solar_inverter", "solar_kw", solar, "kW")
        )
        cur.execute(
            "INSERT INTO sensor_readings (ts, device, metric, value, unit) VALUES (%s, %s, %s, %s, %s)",
            (now, "energy_meter", "load_kw", load, "kW")
        )
        cur.execute(
            "INSERT INTO sensor_readings (ts, device, metric, value, unit) VALUES (%s, %s, %s, %s, %s)",
            (now, "bms", "soc", soc, "%")
        )
        cur.execute(
            "INSERT INTO sensor_readings (ts, device, metric, value, unit) VALUES (%s, %s, %s, %s, %s)",
            (now, "grid_smart_meter", "price", price, "INR")
        )
        
        # Insert agent decision
        cur.execute(
            "INSERT INTO agent_decisions (ts, mode, action, action_name, solar_kw, load_kw, soc, price, safety_status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (now, "rule_based", final_action, final_name, solar, load, soc, price, safety_status)
        )
        
        if final_action == 1:
            soc = min(0.95, soc + 0.04)
        elif final_action == 2:
            soc = max(0.10, soc - 0.03)

    cur.close()
    conn.close()
    print("[DB Feeder] Telemetry populated into TimescaleDB successfully!")

if __name__ == "__main__":
    stream_telemetry_to_timescale()
