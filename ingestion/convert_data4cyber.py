"""
Data4Cyber -> MAEMS converter with proper horizon forecasts and correct SoC envelope.

Fixes:
* `hour` was hardcoded to 13.0 -> now parsed from timestamp.
* `solar_fc_1h` / `load_fc_1h` were +12 steps @ 5 s (= 60 s) -> now true 1-hour horizon (+720 steps).
* SoC clip [0.05, 0.98] vs system [0.10, 0.95] -> aligned to system envelope.
* `grid_status` was hardcoded 1 -> now parsed from `Grid_Connected` column.
* Empty input file IndexError -> guarded.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict, List

import project_paths


def _parse_hour(ts_str: str) -> float:
    """Extract hour (0-23) from an ISO-like timestamp string."""
    if not ts_str:
        return 12.0
    try:
        # Try common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return float(datetime.strptime(ts_str[:19], fmt).hour)
            except ValueError:
                continue
        # Last resort: split on space/T and parse the time part
        time_part = ts_str.split("T")[-1].split(" ")[-1]
        return float(time_part.split(":")[0])
    except Exception:
        return 12.0


def convert_data4cyber_to_maems(
    input_csv: str = "data/raw_data4cyber/data4cyber_dataset/S1_industroyer_pv/dataset.csv",
    output_csv: str = "data/data4cyber_maems_processed.csv",
    horizon_steps: int = 720,  # 720 * 5s = 3600s = 1 hour
    stride: int = 5,           # subsample 1 Hz -> 5 s
    verbose: bool = True,
) -> str:
    if verbose:
        print(f"[Data4Cyber Adapter] Reading {input_csv}...")

    rows: List[Dict[str, Any]] = []
    with open(project_paths.resolve(input_csv), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # PV Generation kW (p_sum3 is in Watts in telemetry)
            pv_w = float(r.get("PV-Inverter-AC-Meter.p_sum3", 0.0) or 0.0)
            pv_kw = max(0.0, round(pv_w / 1000.0, 3))

            # Loads kW (Load A + Load B)
            load_a_w = float(r.get("Load-A-AC-Meter.p_sum3", 0.0) or 0.0)
            load_b_w = float(r.get("Load-B-AC-Meter.p_sum3", 0.0) or 0.0)
            load_kw = max(0.2, round((load_a_w + load_b_w) / 1000.0, 3))

            # Battery SoC (0.0 to 1.0) - align with system envelope [0.10, 0.95]
            soc_val = float(r.get("Battery-SOC-Reference.soc_value", 50.0) or 50.0)
            soc = round(
                min(max(soc_val / 100.0 if soc_val > 1.0 else soc_val, 0.10), 0.95), 4
            )

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

            # Grid status
            grid_connected = r.get("Grid_Connected", "1")
            grid_status = 1 if str(grid_connected).strip().lower() in ("1", "true", "yes") else 0

            # Attack annotation
            attack_str = str(r.get("attack_active", "False")).lower()
            attack_active = 1 if attack_str in ("true", "1") else 0

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
                "grid_status": grid_status,
                "attack_active": attack_active,
            })

    # Subsample (1 Hz -> 5 s default)
    sampled = rows[::stride]

    if not sampled:
        raise ValueError(f"Subsampling produced zero rows (input had {len(rows)} rows, stride={stride})")

    # Calculate true 1-hour forecasts
    for idx in range(len(sampled)):
        future_idx = min(len(sampled) - 1, idx + horizon_steps)
        sampled[idx]["solar_fc_1h"] = sampled[future_idx]["solar_kw"]
        sampled[idx]["load_fc_1h"] = sampled[future_idx]["load_kw"]
        # Hour from timestamp, not hardcoded
        sampled[idx]["hour"] = _parse_hour(sampled[idx]["timestamp"])

    # Ensure output directory
    os.makedirs(os.path.dirname(project_paths.resolve(output_csv)), exist_ok=True)

    # Write in deterministic column order
    fieldnames = [
        "timestamp", "solar_kw", "load_kw", "soc", "batt_temp",
        "grid_import_kw", "grid_voltage", "grid_freq", "price",
        "grid_status", "attack_active", "solar_fc_1h", "load_fc_1h", "hour"
    ]
    with open(project_paths.resolve(output_csv), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    if verbose:
        print(f"[Data4Cyber Adapter] Converted {len(rows)} raw -> {len(sampled)} subsampled records "
              f"(stride={stride}, horizon={horizon_steps*5}s) into {output_csv}")

    return output_csv


if __name__ == "__main__":
    convert_data4cyber_to_maems()