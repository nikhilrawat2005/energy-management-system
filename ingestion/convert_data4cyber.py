import csv
import glob
import os
import numpy as np

def convert_data4cyber_to_maems(input_csv="data/raw_data4cyber/data4cyber_dataset/S0_benign_baseline/dataset.csv", output_csv="data/data4cyber_maems_processed.csv"):
    """
    Parses real industrial telemetry from Data4Cyber:
      - PV Generation: PV-Inverter-AC-Meter.p_sum3 (W -> kW)
      - Microgrid Demand: Load-A + Load-B (W -> kW)
      - Battery Storage: Battery-SOC-Reference.soc_value (0-100% -> 0.0-1.0)
      - BSS Inverter Power: BSS-Inverter-AC-Meter.p_sum3 (W -> kW)
      - Grid / Substation Import: Substation-AC-Meter.p_sum3 (W -> kW)
      - Grid Voltage & Freq: Substation-AC-Meter.uln1, Substation-AC-Meter.freq
      - Dynamic Price: MQTT-Price-Signal.price_value
      - Attack annotations: attack_active
    """
    print(f"[Data4Cyber Adapter] Reading {input_csv}...")
    
    rows = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # PV Generation kW (p_sum3 is in Watts in telemetry)
            pv_w = float(r.get("PV-Inverter-AC-Meter.p_sum3", 0.0) or 0.0)
            pv_kw = max(0.0, round(pv_w / 1000.0, 3))
            
            # Loads kW (Load A + Load B)
            load_a_w = float(r.get("Load-A-AC-Meter.p_sum3", 0.0) or 0.0)
            load_b_w = float(r.get("Load-B-AC-Meter.p_sum3", 0.0) or 0.0)
            load_kw = max(0.2, round((load_a_w + load_b_w) / 1000.0, 3))
            
            # Battery SoC (0.0 to 1.0)
            soc_val = float(r.get("Battery-SOC-Reference.soc_value", 50.0) or 50.0)
            soc = round(float(np.clip(soc_val / 100.0 if soc_val > 1.0 else soc_val, 0.05, 0.98)), 4)
            
            # Substation / Grid
            sub_w = float(r.get("Substation-AC-Meter.p_sum3", 0.0) or 0.0)
            grid_import_kw = round(sub_w / 1000.0, 3)
            
            v_grid = float(r.get("Substation-AC-Meter.uln1", 230.0) or 230.0)
            f_grid = float(r.get("Substation-AC-Meter.freq", 50.0) or 50.0)
            
            # Price signal
            price_raw = r.get("MQTT-Price-Signal.price_value", "7.0")
            price_val = float(price_raw) if price_raw and price_raw.strip() else 7.0
            if price_val <= 0:
                price_val = 7.0
                
            attack_str = str(r.get("attack_active", "False")).lower()
            attack_active = 1 if attack_str in ["true", "1"] else 0
            
            rows.append({
                "timestamp": r.get("timestamp", ""),
                "solar_kw": pv_kw,
                "load_kw": load_kw,
                "soc": soc,
                "batt_temp": 28.5,
                "grid_import_kw": grid_import_kw,
                "grid_voltage": round(v_grid, 1),
                "grid_freq": round(f_grid, 2),
                "price": round(price_val, 2),
                "grid_status": 1,
                "attack_active": attack_active
            })

    # Subsample 1 Hz down to 5s steps for clean multi-agent execution
    sampled = rows[::5]
    
    # Calculate forecasts (t+12 steps ahead = 1 min horizon)
    for idx in range(len(sampled)):
        future_idx = min(len(sampled) - 1, idx + 12)
        sampled[idx]["solar_fc_1h"] = sampled[future_idx]["solar_kw"]
        sampled[idx]["load_fc_1h"] = sampled[future_idx]["load_kw"]
        sampled[idx]["hour"] = 13.0

    header = list(sampled[0].keys())
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(sampled)

    print(f"[Data4Cyber Adapter] Successfully converted {len(sampled)} real telemetry records into {output_csv}")
    return output_csv

if __name__ == "__main__":
    convert_data4cyber_to_maems()
