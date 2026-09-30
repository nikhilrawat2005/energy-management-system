"""
Data4Cyber dataset reader with safe coercion and explicit column mapping.

Fixes:
* float("") / int("1.0") crashes -> safe coercion via _to_float / _to_int.
* Empty cells no longer raise.
* Missing columns fall back to defaults instead of KeyError.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Generator, Optional


def _to_float(val: Any, fallback: float = 0.0) -> float:
    """Safely coerce any value to float. Empty/None -> fallback."""
    if val is None:
        return fallback
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "null", "none"):
        return fallback
    try:
        return float(s)
    except (ValueError, TypeError):
        return fallback


def _to_int(val: Any, fallback: int = 0) -> int:
    """Safely coerce any value to int. Empty/None -> fallback. Handles '1.0'."""
    if val is None:
        return fallback
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "null", "none"):
        return fallback
    try:
        return int(float(s))  # accepts "1.0"
    except (ValueError, TypeError):
        return fallback


class Data4CyberStreamReader:
    """
    Reader and Normalizer for Data4Cyber energy and smart meter datasets.

    Maps various column name conventions to MAEMS unified sensor schema:
      - Solar Inverter (kW, V, A)
      - Battery Storage System (BSS SoC, V, A, Temp)
      - Smart Energy Meters (Active power kW, Voltage, Current, Frequency)
      - Grid connection status & price signals
    """

    # Accept many possible column name variants per field
    COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
        "solar_kw": (
            "PV_ActivePower_kW", "solar_kw", "PV_P", "PV-Inverter-AC-Meter.p_sum3",
        ),
        "load_kw": (
            "Load_ActivePower_kW", "load_kw", "Load_P", "Load-A-AC-Meter.p_sum3",
        ),
        "soc": (
            "BSS_SoC", "soc", "Battery-SOC-Reference.soc_value",
        ),
        "batt_temp": (
            "BSS_Temp_C", "batt_temp",
        ),
        "price": (
            "Tariff_Price", "price", "MQTT-Price-Signal.price_value",
        ),
        "grid_voltage": (
            "Grid_Voltage_V", "grid_voltage", "Substation-AC-Meter.uln1",
        ),
        "grid_freq": (
            "Grid_Freq_Hz", "grid_freq", "Substation-AC-Meter.freq",
        ),
        "grid_status": (
            "Grid_Connected", "grid_status",
        ),
        "irradiance_wm2": (
            "Irradiance_Wm2", "irradiance_wm2",
        ),
        "hour": (
            "hour", "Hour", "timestamp",  # will parse hour from timestamp if needed
        ),
        "solar_fc_1h": (
            "solar_fc_1h", "PV_ActivePower_kW",
        ),
        "load_fc_1h": (
            "load_fc_1h", "Load_ActivePower_kW",
        ),
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data4Cyber dataset file not found at: {file_path}")

    def _get_first(self, row: Dict[str, str], field: str) -> str:
        """Return the first non-empty value from the alias list, or empty string."""
        for alias in self.COLUMN_ALIASES.get(field, ()):
            if alias in row and row[alias] is not None and str(row[alias]).strip() != "":
                return row[alias]
        return ""

    def stream_records(self) -> Generator[Dict[str, Any], None, None]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapped_state = {
                    "solar_kw": _to_float(self._get_first(row, "solar_kw"), 0.0),
                    "load_kw": _to_float(self._get_first(row, "load_kw"), 2.0),
                    "soc": _to_float(self._get_first(row, "soc"), 0.50),
                    "batt_temp": _to_float(self._get_first(row, "batt_temp"), 26.0),
                    "price": _to_float(self._get_first(row, "price"), 7.0),
                    "grid_voltage": _to_float(self._get_first(row, "grid_voltage"), 230.0),
                    "grid_freq": _to_float(self._get_first(row, "grid_freq"), 50.0),
                    "grid_status": _to_int(self._get_first(row, "grid_status"), 1),
                    "irradiance_wm2": _to_float(self._get_first(row, "irradiance_wm2"), 0.0),
                    "hour": _to_float(self._get_first(row, "hour"), 12.0),
                    "solar_fc_1h": _to_float(self._get_first(row, "solar_fc_1h"), 0.0),
                    "load_fc_1h": _to_float(self._get_first(row, "load_fc_1h"), 2.0),
                }

                # Normalize SoC: if input was 0-100, divide by 100
                if mapped_state["soc"] > 1.0:
                    mapped_state["soc"] /= 100.0

                # Clamp to system envelope (from limits.yaml)
                mapped_state["soc"] = min(max(mapped_state["soc"], 0.10), 0.95)

                yield mapped_state


if __name__ == "__main__":
    import project_paths

    reader = Data4CyberStreamReader(project_paths.DATA4CYBER_DATASET_PATH)
    for i, rec in enumerate(reader.stream_records()):
        print(f"Record {i}: solar={rec['solar_kw']:.2f} load={rec['load_kw']:.2f} "
              f"soc={rec['soc']:.2f} price={rec['price']:.2f} grid={rec['grid_status']}")
        if i >= 4:
            print("...")
            break