import numpy as np
import csv
import os
import math
from datetime import datetime, timedelta

def generate_synthetic_energy_dataset(days=30, freq_minutes=15, output_csv="data/dataset_15min.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    total_steps = int((days * 24 * 60) / freq_minutes)
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    
    timestamps = [start_time + timedelta(minutes=i * freq_minutes) for i in range(total_steps)]
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
    day_of_weeks = np.array([ts.weekday() for ts in timestamps])
    is_weekends = np.isin(day_of_weeks, [5, 6]).astype(int)
    
    # 1. Weather: Ambient temp, humidity, irradiance
    base_temp = 25.0 + 8.0 * np.sin(np.pi * (hours - 8.0) / 12.0)
    ambient_temp = np.clip(base_temp + np.random.normal(0, 1.2, total_steps), 10.0, 45.0)
    humidity = np.clip(90.0 - (ambient_temp - 10.0) * 1.8 + np.random.normal(0, 3, total_steps), 25.0, 95.0)
    
    solar_mask = (hours >= 6.0) & (hours <= 18.0)
    solar_peak = 950.0  # W/m2
    irradiance = np.zeros(total_steps)
    
    steps_per_day = 24 * (60 // freq_minutes)
    day_indices = np.arange(total_steps) // steps_per_day
    cloud_factor_per_day = np.random.uniform(0.65, 1.0, size=days + 1)
    cloud_factors = cloud_factor_per_day[day_indices]
    
    solar_angles = np.pi * (hours - 6.0) / 12.0
    irradiance[solar_mask] = (
        solar_peak * np.sin(solar_angles[solar_mask]) * cloud_factors[solar_mask] 
        + np.random.normal(0, 15, np.sum(solar_mask))
    )
    irradiance = np.clip(irradiance, 0.0, 1200.0)
    
    # Solar generation (kW)
    pv_efficiency = 0.18
    pv_area_m2 = 50.0
    solar_kw = np.clip(irradiance * pv_area_m2 * pv_efficiency / 1000.0, 0.0, 10.0)
    
    # 2. Demand Profile
    morning_peak = 3.5 * np.exp(-((hours - 8.5) ** 2) / (2.0 * (1.5 ** 2)))
    evening_peak = 5.0 * np.exp(-((hours - 20.0) ** 2) / (2.0 * (2.0 ** 2)))
    base_load = 1.2 + 0.4 * is_weekends
    hvac_load = np.maximum(0, (ambient_temp - 26.0) * 0.15)
    
    load_kw = base_load + morning_peak + evening_peak + hvac_load + np.random.normal(0, 0.25, total_steps)
    load_kw = np.clip(load_kw, 0.4, 15.0)
    
    # 3. Electricity Tariff (₹/kWh)
    def calc_price(h):
        if (h < 6.0) or (h >= 22.0):
            return 4.5
        elif (h >= 18.0) and (h < 22.0):
            return 11.5
        else:
            return 7.0
            
    prices = np.array([calc_price(h) for h in hours])
    
    # 4. Grid voltage, frequency, status
    grid_voltage = np.clip(230.0 + np.random.normal(0, 4.0, total_steps), 210.0, 250.0)
    grid_freq = np.clip(50.0 + np.random.normal(0, 0.1, total_steps), 49.6, 50.4)
    grid_status = np.where(np.random.rand(total_steps) > 0.005, 1, 0)
    
    # 5. Forecast targets (1h and 4h horizons)
    steps_1h = 60 // freq_minutes
    steps_4h = 4 * steps_1h
    
    solar_fc_1h = np.zeros(total_steps)
    solar_fc_1h[:-steps_1h] = solar_kw[steps_1h:]
    solar_fc_1h[-steps_1h:] = solar_kw[-1]
    solar_fc_1h = np.clip(solar_fc_1h + np.random.normal(0, 0.15, total_steps), 0.0, 10.0)
    
    load_fc_1h = np.zeros(total_steps)
    load_fc_1h[:-steps_1h] = load_kw[steps_1h:]
    load_fc_1h[-steps_1h:] = load_kw[-1]
    load_fc_1h = np.clip(load_fc_1h + np.random.normal(0, 0.2, total_steps), 0.4, 15.0)
    
    solar_fc_4h = np.zeros(total_steps)
    solar_fc_4h[:-steps_4h] = solar_kw[steps_4h:]
    solar_fc_4h[-steps_4h:] = solar_kw[-1]
    solar_fc_4h = np.clip(solar_fc_4h + np.random.normal(0, 0.35, total_steps), 0.0, 10.0)
    
    load_fc_4h = np.zeros(total_steps)
    load_fc_4h[:-steps_4h] = load_kw[steps_4h:]
    load_fc_4h[-steps_4h:] = load_kw[-1]
    load_fc_4h = np.clip(load_fc_4h + np.random.normal(0, 0.4, total_steps), 0.4, 15.0)
    
    batt_temp = ambient_temp + 2.5 + np.random.normal(0, 0.5, total_steps)
    
    header = [
        "timestamp", "hour", "day_of_week", "is_weekend", "ambient_temp", "humidity",
        "irradiance_wm2", "solar_kw", "load_kw", "price", "grid_voltage", "grid_freq",
        "grid_status", "solar_fc_1h", "load_fc_1h", "solar_fc_4h", "load_fc_4h", "batt_temp"
    ]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(total_steps):
            writer.writerow([
                timestamps[i].isoformat(),
                round(float(hours[i]), 3),
                int(day_of_weeks[i]),
                int(is_weekends[i]),
                round(float(ambient_temp[i]), 2),
                round(float(humidity[i]), 2),
                round(float(irradiance[i]), 2),
                round(float(solar_kw[i]), 3),
                round(float(load_kw[i]), 3),
                round(float(prices[i]), 2),
                round(float(grid_voltage[i]), 2),
                round(float(grid_freq[i]), 2),
                int(grid_status[i]),
                round(float(solar_fc_1h[i]), 3),
                round(float(load_fc_1h[i]), 3),
                round(float(solar_fc_4h[i]), 3),
                round(float(load_fc_4h[i]), 3),
                round(float(batt_temp[i]), 2)
            ])
            
    print(f"[Simulator] Generated {total_steps} rows directly saved to {output_csv}")
    return output_csv

if __name__ == "__main__":
    generate_synthetic_energy_dataset()
