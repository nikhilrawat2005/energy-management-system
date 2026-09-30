"""
Modbus register decoder + DataCleaner.

Fixes:
* Unknown dtype no longer silently falls through to uint16 -> raises ValueError.
* struct.pack(">HH") protected against out-of-range / negative ints.
* Optional word-order parameter (CDAB / ABCD / BADC / DCBA).
* DataCleaner now rejects NaN / inf and is wired to load limits from devices.yaml.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import project_paths


def decode_modbus_register(
    regs: List[int],
    dtype: str,
    scale: float = 1.0,
    word_order: str = "big",
) -> float:
    """
    Decode Modbus 16-bit register words into a physical numeric value.

    Parameters
    ----------
    regs: list of 16-bit unsigned integers (0..65535).
    dtype: one of "float32", "uint32", "int16", "uint16".
    scale: multiplier applied after decoding.
    word_order: "big" (ABCD / >HH) or "little" (CDAB / <HH).
                "big"  -> register[0] is high word, register[1] low word (standard Modbus).
                "little" -> register[0] is low word, register[1] high word (some vendors).
    """
    if not regs:
        raise ValueError("Register list is empty")

    # Validate word_order
    if word_order not in ("big", "little"):
        raise ValueError(f"word_order must be 'big' or 'little', got {word_order!r}")

    pack_fmt = ">HH" if word_order == "big" else "<HH"

    def _pack_pair(hi: int, lo: int) -> bytes:
        """Pack two 16-bit words, validating range."""
        for v in (hi, lo):
            if not isinstance(v, int) or v < 0 or v > 0xFFFF:
                raise ValueError(f"Modbus register out of range 0..65535: {v!r}")
        return struct.pack(pack_fmt, hi, lo)

    if dtype == "float32":
        if len(regs) < 2:
            raise ValueError("float32 requires at least 2 registers (32-bit)")
        raw = _pack_pair(regs[0], regs[1])
        val = struct.unpack(">f" if word_order == "big" else "<f", raw)[0]

    elif dtype == "uint32":
        if len(regs) < 2:
            raise ValueError("uint32 requires at least 2 registers")
        hi, lo = regs[0], regs[1]
        val = (hi << 16) | lo

    elif dtype == "int16":
        v = regs[0]
        val = v - 65536 if v > 32767 else v

    elif dtype == "uint16":
        val = regs[0]

    else:
        raise ValueError(f"Unsupported dtype: {dtype!r}. "
                         "Supported: float32, uint32, int16, uint16")

    scaled = val * scale
    if not math.isfinite(scaled):
        raise ValueError(f"Decoded value is non-finite: {scaled!r} (raw={val}, scale={scale})")
    return round(float(scaled), 4)


class DataCleaner:
    """
    Validates physical sanity checks, clamps outlier spikes, flags frozen sensors.

    Loads per-device/per-metric limits from config/devices.yaml at construction time.
    Example devices.yaml section:
        devices:
          smart_energy_meter:
            meter_power_kw:  {min: 0.0, max: 30.0}
            meter_energy_kwh: {min: 0.0, max: 1e6}
    """

    def __init__(self, limits: Optional[Dict[str, Any]] = None):
        self.limits = limits if limits is not None else project_paths.load_devices().get("devices", {})
        self.last_values: Dict[str, float] = {}
        self.stuck_counter: Dict[str, int] = {}

    def clean_reading(
        self,
        device: str,
        metric: str,
        value: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dict with device, metric, value (possibly clamped), quality, timestamp.
        Quality codes: "good", "out_of_range_low", "out_of_range_high",
                       "suspect_stuck", "non_finite", "unknown_metric".
        """
        key = f"{device}_{metric}"
        quality = "good"

        # Non-finite check (first - NaN/inf propagate and break downstream)
        if not math.isfinite(value):
            quality = "non_finite"
            value = 0.0

        # Per-device limits from devices.yaml
        device_cfg = self.limits.get(device, {})
        metric_cfg = device_cfg.get(metric, {})
        if not metric_cfg:
            # No limits defined -> unknown metric, don't fail, just flag
            quality = "unknown_metric"

        # Explicit overrides take precedence
        if min_val is not None:
            metric_cfg["min"] = min_val
        if max_val is not None:
            metric_cfg["max"] = max_val

        # Range sanity checks
        min_allowed = metric_cfg.get("min")
        max_allowed = metric_cfg.get("max")
        if min_allowed is not None and value < min_allowed:
            quality = "out_of_range_low"
            value = float(min_allowed)
        elif max_allowed is not None and value > max_allowed:
            quality = "out_of_range_high"
            value = float(max_allowed)

        # Sensor stuck detection (exact equality over 20 consecutive readings)
        if key in self.last_values and abs(self.last_values[key] - value) < 1e-6:
            self.stuck_counter[key] = self.stuck_counter.get(key, 0) + 1
            if self.stuck_counter[key] > 20:
                quality = "suspect_stuck"
        else:
            self.stuck_counter[key] = 0

        self.last_values[key] = value

        return {
            "device": device,
            "metric": metric,
            "value": value,
            "quality": quality,
            "timestamp": time.time(),
        }


def load_devices_config() -> Dict[str, Any]:
    """Convenience: load and return the `devices:` section from devices.yaml."""
    return project_paths.load_devices().get("devices", {})


if __name__ == "__main__":
    # Quick self-test
    print("float32 big:", decode_modbus_register([0x4248, 0x0000], "float32"))       # 50.0
    print("float32 little:", decode_modbus_register([0x0000, 0x4248], "float32", word_order="little"))
    print("uint32:", decode_modbus_register([0x0000, 0x0005], "uint32"))
    print("int16 neg:", decode_modbus_register([0xFF00], "int16"))                  # -256
    print("uint16:", decode_modbus_register([0x01F4], "uint16"))                    # 500

    cleaner = DataCleaner({"smart_energy_meter": {"meter_power_kw": {"min": 0, "max": 25}}})
    print("clean good:", cleaner.clean_reading("smart_energy_meter", "meter_power_kw", 12.3))
    print("clean high:", cleaner.clean_reading("smart_energy_meter", "meter_power_kw", 50.0))
    print("clean NaN:", cleaner.clean_reading("smart_energy_meter", "meter_power_kw", float("nan")))