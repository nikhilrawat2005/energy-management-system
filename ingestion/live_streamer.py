import time
import psycopg2
import csv
from datetime import datetime, timezone
from coordinator.coordinator import MultiAgentCoordinator
from forecasting.features import SimpleGradientBoostingRegressor
from ingestion.api_fetcher import WeatherAPIFetcher
import numpy as np

def run_unified_streamer(dataset_path="data/dataset_15min.csv", interval_sec=2.0):
    """
    Unified Live Streamer executing the complete 13-sensor architecture:
    1. Smart Energy Meter (kW, kWh, V, A, Hz, PF)
    2. CT (Current A)
    3. Voltage Sensor / PT (Voltage V)
    4. Solar Inverter (kW, kWh, V, A)
    5. Pyranometer (Irradiance W/m²)
    6. BMS (SoC %, V, A, °C)
    7. NTC Thermistor (Battery Temp °C)
    8. DHT22 / PT100 (Ambient Temp °C)
    9. Humidity Sensor (%RH)
    10. Anemometer (Wind Speed m/s)
    11. Weather API / Station (Live Open-Meteo Cloud %, Temp, Irradiance)
    12. Electricity Tariff API (₹/kWh dynamic ToD)
    13. Grid Smart Meter (kW, kWh, V, A, Hz, PF, ON/OFF)
    
    Plus:
    - ML Forecasts (Solar 1h & Demand 1h)
    - Multi-Agent Orchestration (Solar, Demand, Battery, Price)
    - Safety Interlock & Physical Hardware Protection
    """
    print("[Unified Streamer] Connecting to TimescaleDB (port 5434)...")
    conn = psycopg2.connect(
        host="localhost",
        port=5434,
        user="maems_user",
        password="maems_password",
        dbname="maems_energy"
    )
    conn.autocommit = True
    cur = conn.cursor()
    print("[Unified Streamer] Connected to Database successfully!")

    coord = MultiAgentCoordinator()
    weather_fetcher = WeatherAPIFetcher()
    
    demand_model = SimpleGradientBoostingRegressor()
    solar_model = SimpleGradientBoostingRegressor()
    demand_model.load("models/demand_forecast.json")
    solar_model.load("models/solar_forecast.json")
    print("[Unified Streamer] Loaded ML Models & Live Weather API Fetcher!")

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    soc = 0.50
    step_idx = 0
    total_savings_inr = 0.0
    accumulated_solar_kwh = 12.4
    accumulated_load_kwh = 18.2
    accumulated_grid_kwh = 8.5
    last_weather_fetch = 0
    cached_weather = {"ambient_temp": 29.8, "humidity": 60, "irradiance_wm2": 650.0, "cloud_cover_pct": 10.0}

    print(f"[Unified Streamer] Streaming all 13 sensors + weather + forecasts every {interval_sec}s...")

    while True:
        r = rows[step_idx % len(rows)]
        now = datetime.now(timezone.utc)
        
        # Periodic live weather poll (every 30 seconds)
        if time.time() - last_weather_fetch > 30:
            live_w = weather_fetcher.fetch_current_and_forecast()
            if live_w.get("status") == "online":
                cached_weather = live_w
            last_weather_fetch = time.time()

        hour = float(r["hour"])
        day_of_week = float(r["day_of_week"])
        is_weekend = float(r["is_weekend"])

        # Sensor 5 & 11: Pyranometer & Weather API Irradiance
        irradiance_wm2 = float(cached_weather.get("irradiance_wm2", r["irradiance_wm2"]))
        cloud_cover_pct = float(cached_weather.get("cloud_cover_pct", 15.0))
        
        # Sensor 8 & 9: DHT22 / PT100 Ambient Temp & Humidity
        ambient_temp_c = float(cached_weather.get("ambient_temp", r["ambient_temp"]))
        humidity_pct = float(cached_weather.get("humidity", r["humidity"]))
        
        # Sensor 10: Anemometer Wind Speed (m/s)
        wind_speed_ms = round(float(2.5 + 1.8 * np.sin(hour / 3.0) + np.random.normal(0, 0.4)), 2)
        wind_speed_ms = max(0.2, wind_speed_ms)

        # Sensor 4: Solar Inverter (kW, V, A, kWh)
        solar_kw = float(r["solar_kw"])
        solar_v = round(380.0 + np.random.normal(0, 3.0), 1) if solar_kw > 0.05 else 0.0
        solar_a = round((solar_kw * 1000.0) / max(1.0, solar_v), 2)
        accumulated_solar_kwh += (solar_kw * (interval_sec / 3600.0))

        # Sensor 1, 2, 3: Smart Energy Meter, CT, Voltage Sensor/PT
        load_kw = float(r["load_kw"])
        grid_v = round(float(r["grid_voltage"]), 1)
        grid_f = round(float(r["grid_freq"]), 2)
        grid_pf = round(float(0.96 + np.random.normal(0, 0.01)), 3)
        load_a = round((load_kw * 1000.0) / (grid_v * grid_pf), 2)
        accumulated_load_kwh += (load_kw * (interval_sec / 3600.0))

        # Sensor 7: NTC Thermistor Battery Temp (°C)
        batt_temp_c = round(float(ambient_temp_c + 2.0 + (abs(load_kw - solar_kw) * 0.4) + np.random.normal(0, 0.2)), 2)

        # Sensor 12: Electricity Tariff (₹/kWh)
        price = float(r["price"])
        grid_status = int(r["grid_status"])

        # ML Horizon Predictions
        demand_feats = np.array([[hour, day_of_week, is_weekend, ambient_temp_c, humidity_pct, load_kw, load_kw]], dtype=np.float32)
        solar_feats = np.array([[hour, irradiance_wm2, ambient_temp_c, solar_kw, solar_kw]], dtype=np.float32)
        pred_demand_1h = max(0.4, float(demand_model.predict(demand_feats)[0]))
        pred_solar_1h = max(0.0, float(solar_model.predict(solar_feats)[0]))

        # Coordinator Tick
        r_state = {
            "solar_kw": solar_kw,
            "load_kw": load_kw,
            "soc": soc,
            "batt_temp": batt_temp_c,
            "price": price,
            "grid_status": grid_status,
            "grid_voltage": grid_v,
            "grid_freq": grid_f,
            "hour": hour,
            "irradiance_wm2": irradiance_wm2,
            "solar_fc_1h": pred_solar_1h,
            "load_fc_1h": pred_demand_1h
        }
        res = coord.process_tick(r_state, mode="rule_based", current_time_sec=time.time())
        final_action = res["final_action"]
        final_name = res["final_action_name"]
        safety_status = res["safety_status"]

        # Battery dynamics & flow calculations
        # Sensor 6: BMS (SoC %, V, A, °C)
        batt_v = round(48.0 + (soc * 6.0), 2) # 48V to 54V pack
        batt_kw = 0.0
        
        if final_action == 1: # Charge Solar
            surplus = max(0.0, solar_kw - load_kw)
            batt_kw = min(5.0, surplus, (0.95 - soc) * 15.0 / 0.25)
            soc = min(0.95, soc + (batt_kw * 0.95 * 0.25) / 15.0)
            actual_grid_import = max(0.0, load_kw - min(solar_kw, load_kw))
        elif final_action == 2: # Discharge Battery
            deficit = max(0.0, load_kw - solar_kw)
            batt_kw = -min(5.0, deficit, (soc - 0.10) * 15.0 / 0.25)
            soc = max(0.10, soc - (abs(batt_kw) / 0.95 * 0.25) / 15.0)
            actual_grid_import = max(0.0, load_kw - (min(solar_kw, load_kw) + abs(batt_kw)))
        elif final_action == 3 and grid_status == 1: # Charge Grid
            batt_kw = min(5.0, (0.95 - soc) * 15.0 / 0.25)
            soc = min(0.95, soc + (batt_kw * 0.95 * 0.25) / 15.0)
            actual_grid_import = load_kw + batt_kw
        else:
            actual_grid_import = max(0.0, load_kw - min(solar_kw, load_kw))

        batt_a = round((batt_kw * 1000.0) / batt_v, 2)
        accumulated_grid_kwh += (actual_grid_import * (interval_sec / 3600.0))

        # Sensor 13: Grid Smart Meter (kW, V, A, Hz, PF, ON/OFF)
        grid_a = round((actual_grid_import * 1000.0) / (grid_v * grid_pf), 2) if grid_status == 1 else 0.0
        
        # Savings
        grid_only_cost = load_kw * price * 0.25
        maems_cost = actual_grid_import * price * 0.25
        total_savings_inr += max(0.0, grid_only_cost - maems_cost)

        # INSERT READINGS FOR ALL 13 SENSORS
        all_sensor_records = [
            # 1. Smart Energy Meter
            ("smart_energy_meter", "meter_power_kw", load_kw, "kW"),
            ("smart_energy_meter", "meter_energy_kwh", round(accumulated_load_kwh, 2), "kWh"),
            ("smart_energy_meter", "meter_pf", grid_pf, ""),
            # 2. CT (Current Transformer)
            ("ct_sensor", "ct_current_a", load_a, "A"),
            # 3. Voltage Sensor / PT
            ("pt_voltage_sensor", "pt_voltage_v", grid_v, "V"),
            # 4. Solar Inverter
            ("solar_inverter", "solar_kw", solar_kw, "kW"),
            ("solar_inverter", "solar_yield_kwh", round(accumulated_solar_kwh, 2), "kWh"),
            ("solar_inverter", "solar_voltage_v", solar_v, "V"),
            ("solar_inverter", "solar_current_a", solar_a, "A"),
            # 5. Pyranometer
            ("pyranometer", "solar_irradiance", irradiance_wm2, "W/m²"),
            # 6. BMS
            ("bms", "soc", soc, "%"),
            ("bms", "bms_voltage_v", batt_v, "V"),
            ("bms", "bms_current_a", batt_a, "A"),
            ("bms", "bms_power_kw", round(batt_kw, 2), "kW"),
            # 7. NTC Thermistor
            ("ntc_thermistor", "batt_temp_c", batt_temp_c, "°C"),
            # 8. DHT22 / PT100
            ("dht22_sensor", "ambient_temp_c", ambient_temp_c, "°C"),
            # 9. Humidity Sensor
            ("humidity_sensor", "humidity_pct", humidity_pct, "%RH"),
            # 10. Anemometer
            ("anemometer", "wind_speed_ms", wind_speed_ms, "m/s"),
            # 11. Weather API / Station
            ("weather_api", "cloud_cover_pct", cloud_cover_pct, "%"),
            # 12. Electricity Tariff API
            ("tariff_api", "price", price, "₹/kWh"),
            # 13. Grid Smart Meter
            ("grid_smart_meter", "grid_power_kw", round(actual_grid_import, 2), "kW"),
            ("grid_smart_meter", "grid_freq_hz", grid_f, "Hz"),
            ("grid_smart_meter", "grid_status", grid_status, "ON/OFF"),
            ("grid_smart_meter", "grid_current_a", grid_a, "A"),
            ("grid_smart_meter", "grid_energy_kwh", round(accumulated_grid_kwh, 2), "kWh"),
            # ML Model Forecasts & Economics
            ("ml_forecaster", "pred_solar_1h", pred_solar_1h, "kW"),
            ("ml_forecaster", "pred_demand_1h", pred_demand_1h, "kW"),
            ("economic_calculator", "total_savings_inr", round(total_savings_inr, 2), "INR")
        ]

        for dev, metric, val, unit in all_sensor_records:
            cur.execute(
                "INSERT INTO sensor_readings (ts, device, metric, value, unit) VALUES (%s, %s, %s, %s, %s)",
                (now, dev, metric, val, unit)
            )

        cur.execute(
            "INSERT INTO agent_decisions (ts, mode, action, action_name, solar_kw, load_kw, soc, price, safety_status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (now, "rule_based", final_action, final_name, solar_kw, load_kw, soc, price, safety_status)
        )

        step_idx += 1
        time.sleep(interval_sec)

if __name__ == "__main__":
    run_unified_streamer()
