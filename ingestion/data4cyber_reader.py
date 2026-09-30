import csv
import json
import os
from typing import Dict, Any, Generator

class Data4CyberStreamReader:
    """
    Reader and Normalizer for Data4Cyber energy and smart meter datasets.
    Maps Modbus registers / OT telemetry to MAEMS unified sensor schema:
      - Solar Inverter (kW, V, A)
      - Battery Storage System (BSS SoC, V, A, Temp)
      - Smart Energy Meters (Active power kW, Voltage, Current, Frequency)
      - Grid connection status & price signals
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data4Cyber dataset file not found at: {file_path}")

    def stream_records(self) -> Generator[Dict[str, Any], None, None]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Map various standard Data4Cyber / DER telemetry columns
                mapped_state = {
                    "solar_kw": float(row.get("PV_ActivePower_kW", row.get("solar_kw", row.get("PV_P", 0.0)))),
                    "load_kw": float(row.get("Load_ActivePower_kW", row.get("load_kw", row.get("Load_P", 2.0)))),
                    "soc": float(row.get("BSS_SoC", row.get("soc", 0.50))),
                    "batt_temp": float(row.get("BSS_Temp_C", row.get("batt_temp", 26.0))),
                    "price": float(row.get("Tariff_Price", row.get("price", 7.0))),
                    "grid_voltage": float(row.get("Grid_Voltage_V", row.get("grid_voltage", 230.0))),
                    "grid_freq": float(row.get("Grid_Freq_Hz", row.get("grid_freq", 50.0))),
                    "grid_status": int(row.get("Grid_Connected", row.get("grid_status", 1))),
                    "irradiance_wm2": float(row.get("Irradiance_Wm2", row.get("irradiance_wm2", 0.0))),
                    "hour": float(row.get("hour", 12.0)),
                    "solar_fc_1h": float(row.get("solar_fc_1h", row.get("PV_ActivePower_kW", 0.0))),
                    "load_fc_1h": float(row.get("load_fc_1h", row.get("Load_ActivePower_kW", 2.0))),
                }
                
                # Normalization: ensure SoC is in 0.0 - 1.0 range
                if mapped_state["soc"] > 1.0:
                    mapped_state["soc"] /= 100.0
                    
                yield mapped_state
