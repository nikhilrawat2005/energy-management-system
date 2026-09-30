import struct
import json
import time
from typing import Dict, Any, List, Optional

def decode_modbus_register(regs: List[int], dtype: str, scale: float = 1.0) -> float:
    """
    Decodes Modbus 16-bit register words into physical numeric values.
    Supports float32 (IEEE 754), uint32, int16, and uint16.
    """
    if not regs:
        raise ValueError("Register list is empty")
        
    if dtype == "float32":
        if len(regs) < 2:
            raise ValueError("float32 requires at least 2 registers (32-bit)")
        # Big-endian word order
        raw = struct.pack(">HH", regs[0], regs[1])
        val = struct.unpack(">f", raw)[0]
        return round(float(val * scale), 4)
        
    elif dtype == "uint32":
        if len(regs) < 2:
            raise ValueError("uint32 requires at least 2 registers")
        val = (regs[0] << 16) | regs[1]
        return round(float(val * scale), 4)
        
    elif dtype == "int16":
        v = regs[0] - 65536 if regs[0] > 32767 else regs[0]
        return round(float(v * scale), 4)
        
    elif dtype == "uint16":
        return round(float(regs[0] * scale), 4)
        
    else:
        return round(float(regs[0] * scale), 4)

class DataCleaner:
    """
    Validates physical sanity checks, clamps outlier spikes, flags frozen sensors.
    """
    def __init__(self, limits: Dict[str, Any]):
        self.limits = limits
        self.last_values = {}
        self.stuck_counter = {}

    def clean_reading(self, device: str, metric: str, value: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Dict[str, Any]:
        key = f"{device}_{metric}"
        quality = "good"
        
        # Range sanity checks
        if min_val is not None and value < min_val:
            quality = "out_of_range_low"
            value = min_val
        elif max_val is not None and value > max_val:
            quality = "out_of_range_high"
            value = max_val
            
        # Sensor stuck detection
        if key in self.last_values and abs(self.last_values[key] - value) < 1e-6:
            self.stuck_counter[key] = self.stuck_counter.get(key, 0) + 1
            if self.stuck_counter[key] > 20: # 20 consecutive identical readings
                quality = "suspect_stuck"
        else:
            self.stuck_counter[key] = 0
            
        self.last_values[key] = value
        
        return {
            "device": device,
            "metric": metric,
            "value": value,
            "quality": quality,
            "timestamp": time.time()
        }
