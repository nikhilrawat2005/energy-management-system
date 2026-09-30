# Multi-Agent Energy Management System (MAEMS)
### Real-time data decoding, forecasting, specialized agents and a reinforcement-learning coordinator

---

## 1. Objective

Build a system that reads live data from solar, battery, meters and grid, predicts demand and solar generation, and automatically decides the cheapest and cleanest way to serve the load.

**Goals:** reduce electricity cost, minimize wasted solar energy, maximize renewable use, keep supply reliable and the battery healthy.

---

## 2. System Architecture

```
 SENSORS / DEVICES            DATA LAYER              INTELLIGENCE                 ACTION
 ─────────────────     ──────────────────────    ─────────────────────────    ───────────────
 Energy Meter  ─┐
 Grid Meter    ─┤ Modbus   ┌──────────────┐     ┌──────────────────────┐
 Solar Inverter─┤ RS485 →  │ Decoder      │     │ Forecast Models      │
 BMS           ─┘          │ (raw→values) │     │ - Demand (t+1..24h)  │
 CT/PT/NTC/DHT22/PT100 →   │ + Cleaning   │ →   │ - Solar  (t+1..24h)  │
   ESP32 (ADC/I2C)         └──────┬───────┘     └──────────┬───────────┘
 Pyranometer/Anemometer →         │ MQTT                   │
 Weather API / Tariff API → HTTP  ▼                        ▼
                           ┌──────────────┐     ┌──────────────────────┐    ┌──────────────┐
                           │ TimescaleDB/ │     │ Agents: Solar, Demand│    │ Controller   │
                           │ InfluxDB     │ ──→ │ Battery, Price       │ →  │ (inverter/   │
                           └──────────────┘     └──────────┬───────────┘    │ BMS / relay) │
                                  │                        ▼                └──────────────┘
                                  ▼             ┌──────────────────────┐
                              Grafana           │ Coordinator + RL     │
                             Dashboard          │ (PPO) + safety layer │
                                                └──────────────────────┘
```

---

## 3. Sensors and Data Sources

| # | Source | Measures | Interface | Sampling | Used by |
|---|---|---|---|---|---|
| 1 | Smart Energy Meter | kW, kWh, V, A, Hz, PF | Modbus RTU/TCP | 5 s | Demand |
| 2 | CT | Current (A) | ADC via ESP32 (or meter) | 1 s | Demand |
| 3 | Voltage sensor / PT | Voltage (V) | ADC via ESP32 | 1 s | Demand, Safety |
| 4 | Solar Inverter | kW, kWh, V, A | Modbus | 5 s | Solar |
| 5 | Pyranometer | Irradiance W/m² | Analog / Modbus | 10 s | Solar, forecast |
| 6 | BMS | SoC %, V, A, °C | CAN / Modbus | 5 s | Battery |
| 7 | NTC thermistor | Battery temp | ADC | 10 s | Battery |
| 8 | DHT22 / PT100 | Ambient temp, RH | GPIO / I2C | 30 s | Demand, forecast |
| 9 | Humidity sensor | %RH | GPIO / I2C | 30 s | Forecast |
| 10 | Anemometer | Wind m/s | Pulse counter | 10 s | Forecast |
| 11 | Weather API | Forecast temp, cloud, irradiance | HTTP | 15 min | Forecast |
| 12 | Tariff API / schedule | ₹/kWh | HTTP / static ToD table | 15 min | Price |
| 13 | Grid Smart Meter | kW, kWh, V, A, Hz, PF, ON/OFF | Modbus | 5 s | Price, Safety |

**Unified record:** `timestamp, device_id, metric, value, unit, quality`

---

## 4. Technology Stack

| Layer | Tools |
|---|---|
| Edge / acquisition | ESP32 + ADS1115 ADC, Raspberry Pi or PC gateway, `pymodbus`, `python-can` |
| Transport | MQTT (Mosquitto) |
| Storage | TimescaleDB (PostgreSQL) or InfluxDB |
| Forecasting | Python, pandas, LightGBM / XGBoost, PyTorch (LSTM optional) |
| Agents + coordinator | Python classes, asyncio, JSON messages |
| RL | Gymnasium, Stable-Baselines3 (PPO) |
| Backend API | FastAPI |
| Dashboard | Grafana (+ optional Next.js UI) |
| Deployment | Docker Compose |

---

## 5. Project Structure

```
maems/
├── docker-compose.yml
├── config/
│   ├── devices.yaml            # Modbus map: slave id, registers, scale, type
│   └── limits.yaml             # SoC min/max, max charge kW, temp limits
├── ingestion/
│   ├── modbus_reader.py        # polls devices, decodes registers
│   ├── esp32_firmware/         # ADC sensors → MQTT
│   ├── api_fetcher.py          # weather + tariff
│   └── mqtt_to_db.py
├── forecasting/
│   ├── features.py
│   ├── train_demand.py
│   ├── train_solar.py
│   └── predict.py
├── agents/
│   ├── base.py
│   ├── solar_agent.py
│   ├── demand_agent.py
│   ├── battery_agent.py
│   └── price_agent.py
├── coordinator/
│   ├── coordinator.py
│   ├── rl_env.py               # Gymnasium environment
│   ├── train_rl.py
│   ├── baseline_rules.py
│   └── safety.py
├── simulator/
│   └── generate_data.py        # synthetic data if hardware not ready
├── api/main.py                 # FastAPI
└── notebooks/evaluation.ipynb
```

---

## 6. Data Layer: Decoding Raw Data

Modbus devices return 16-bit registers. Decoding = combine registers by data type, apply scale.

**config/devices.yaml (example)**
```yaml
energy_meter:
  port: /dev/ttyUSB0
  baud: 9600
  slave: 1
  registers:
    voltage: {addr: 0,  type: float32, unit: V}
    current: {addr: 6,  type: float32, unit: A}
    power_kw: {addr: 12, type: float32, unit: kW}
    pf:      {addr: 30, type: float32, unit: ""}
    freq:    {addr: 70, type: float32, unit: Hz}
    energy_kwh: {addr: 342, type: float32, unit: kWh}
solar_inverter:
  port: /dev/ttyUSB1
  slave: 2
  registers:
    power_kw: {addr: 3005, type: uint32, scale: 0.001, unit: kW}
```
> Register addresses are examples. Take the real ones from each device's datasheet.

**ingestion/modbus_reader.py**
```python
import struct, time, json, yaml
import paho.mqtt.client as mqtt
from pymodbus.client import ModbusSerialClient

cfg = yaml.safe_load(open("config/devices.yaml"))
mq = mqtt.Client(); mq.connect("localhost", 1883)

def decode(regs, dtype, scale=1.0):
    if dtype == "float32":
        raw = struct.pack(">HH", regs[0], regs[1])
        return struct.unpack(">f", raw)[0] * scale
    if dtype == "uint32":
        return ((regs[0] << 16) | regs[1]) * scale
    if dtype == "int16":
        v = regs[0] - 65536 if regs[0] > 32767 else regs[0]
        return v * scale
    return regs[0] * scale

def poll(name, dev):
    c = ModbusSerialClient(port=dev["port"], baudrate=dev.get("baud", 9600))
    c.connect()
    for metric, r in dev["registers"].items():
        count = 2 if r["type"] in ("float32", "uint32") else 1
        rr = c.read_holding_registers(r["addr"], count=count, slave=dev["slave"])
        if rr.isError():
            continue
        val = decode(rr.registers, r["type"], r.get("scale", 1.0))
        mq.publish(f"maems/{name}/{metric}",
                   json.dumps({"ts": time.time(), "value": val, "unit": r["unit"]}))
    c.close()

while True:
    for name, dev in cfg.items():
        poll(name, dev)
    time.sleep(5)
```

**Cleaning rules:** drop out-of-range values (e.g. V < 0 or > 300), forward-fill gaps up to 30 s, flag sensor-stuck (no change for N minutes), resample everything to 1-minute and 15-minute grids.

**Table**
```sql
CREATE TABLE readings (
  ts TIMESTAMPTZ NOT NULL, device TEXT, metric TEXT,
  value DOUBLE PRECISION, unit TEXT
);
SELECT create_hypertable('readings', 'ts');
```

---

## 7. Forecasting Module

**Targets:** load (kW) and solar generation (kW), horizon 1 to 24 hours at 15-minute steps.

**Features**
- Demand: last 1h, 24h, 7d same-hour load, hour, weekday, holiday, ambient temp, humidity.
- Solar: pyranometer irradiance, inverter kW lags, cloud cover forecast, forecast irradiance, temperature, hour-of-day, sun elevation.

**Model:** LightGBM per horizon (fast, strong baseline). Upgrade to LSTM if data exceeds a few months.

```python
import lightgbm as lgb, pandas as pd
from sklearn.metrics import mean_absolute_error

df = pd.read_parquet("data/features.parquet")
FEATS = [c for c in df.columns if c not in ("ts", "target")]
split = int(len(df) * 0.8)
tr, te = df.iloc[:split], df.iloc[split:]   # time-based split, never random

m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63)
m.fit(tr[FEATS], tr["target"])
print("MAE:", mean_absolute_error(te["target"], m.predict(te[FEATS])))
m.booster_.save_model("models/demand_h1.txt")
```
**Metrics:** MAE, RMSE, MAPE. Compare to the naive baseline "same time yesterday".

---

## 8. Specialized Agents

Each agent reads current data + forecasts and publishes a structured report to the coordinator.

| Agent | Looks at | Reports |
|---|---|---|
| **Solar** | inverter kW, irradiance, solar forecast | `solar_now`, `solar_next_4h`, `surplus_expected` |
| **Demand** | meter kW, load forecast, temp | `load_now`, `load_next_4h`, `peak_risk` |
| **Battery** | SoC, V, A, battery temp | `soc`, `available_charge_kW`, `available_discharge_kW`, `health_flag` |
| **Price** | tariff, grid status | `price_now`, `price_next_4h`, `cheap_window`, `grid_ok` |

```python
# agents/battery_agent.py
class BatteryAgent:
    def __init__(self, cfg):
        self.cap = cfg["capacity_kwh"]; self.smin = cfg["soc_min"]; self.smax = cfg["soc_max"]
        self.pmax = cfg["max_power_kw"]; self.tmax = cfg["temp_max_c"]

    def report(self, s):
        hot = s["batt_temp"] > self.tmax
        can_dis = 0 if (s["soc"] <= self.smin or hot) else self.pmax
        can_chg = 0 if (s["soc"] >= self.smax or hot) else self.pmax
        return {"agent": "battery", "soc": s["soc"],
                "avail_discharge_kw": can_dis, "avail_charge_kw": can_chg,
                "health_flag": "hot" if hot else "ok"}
```

---

## 9. Coordinator and Reinforcement Learning

### 9.1 Formulation

- **State (9 values):** solar_kw, load_kw, soc, batt_temp, price, solar_forecast_1h, load_forecast_1h, sin(hour), cos(hour)
- **Actions (discrete 4):**
  0. Solar first, grid covers the rest, battery idle
  1. Charge battery from solar surplus
  2. Discharge battery to serve load
  3. Charge battery from grid (only when cheap)
- **Reward per step:**
  `reward = -(grid_cost) - 0.5 * curtailed_solar_kWh - battery_wear_penalty - blackout_penalty + 0.1 * solar_used_kWh`

### 9.2 Environment

```python
# coordinator/rl_env.py
import gymnasium as gym, numpy as np
from gymnasium import spaces

class EnergyEnv(gym.Env):
    def __init__(self, df, cap=10.0, pmax=3.0, eff=0.95, dt=0.25):
        self.df, self.cap, self.pmax, self.eff, self.dt = df, cap, pmax, eff, dt
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i, self.soc = 0, 0.5
        return self._obs(), {}

    def _obs(self):
        r = self.df.iloc[self.i]
        h = r.hour
        return np.array([r.solar, r.load, self.soc, r.batt_temp, r.price,
                         r.solar_fc, r.load_fc,
                         np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)], dtype=np.float32)

    def step(self, a):
        r = self.df.iloc[self.i]
        solar, load, price = r.solar, r.load, r.price
        batt = 0.0                                   # +charge / -discharge (kW)
        if a == 1:   batt = min(self.pmax, max(solar - load, 0), (1 - self.soc) * self.cap / self.dt)
        elif a == 2: batt = -min(self.pmax, self.soc * self.cap / self.dt)
        elif a == 3: batt = min(self.pmax, (1 - self.soc) * self.cap / self.dt)

        self.soc += (batt * self.eff if batt > 0 else batt / self.eff) * self.dt / self.cap
        self.soc = float(np.clip(self.soc, 0.1, 0.95))

        supply = solar - batt if batt > 0 else solar - batt   # solar + discharge - charge
        net = load - supply                                    # >0 import, <0 surplus
        grid_import = max(net, 0); curtailed = max(-net, 0)
        cost = grid_import * price * self.dt
        wear = 0.02 * abs(batt) * self.dt
        reward = -cost - 0.5 * curtailed * self.dt - wear

        self.i += 1
        done = self.i >= len(self.df) - 1
        info = {"cost": cost, "curtailed": curtailed, "grid": grid_import}
        return self._obs(), reward, done, False, info
```

### 9.3 Training

```python
# coordinator/train_rl.py
import pandas as pd
from stable_baselines3 import PPO
from rl_env import EnergyEnv

df = pd.read_parquet("data/rl_train.parquet")   # real logs or simulator output
env = EnergyEnv(df)
model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, gamma=0.99, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("models/ppo_energy")
```

### 9.4 Safety layer (mandatory)

The RL action is never sent directly. `safety.py` overrides it when:
- SoC < 10% or > 95%, or battery temp above limit → block discharge/charge.
- Grid OFF → force islanded mode (solar + battery only, shed non-critical load).
- Voltage or frequency out of range → stop battery power flow.
- Command rate limited to once per 15 minutes to avoid relay wear.

### 9.5 Fallback

If the model, sensors or forecasts fail, switch to `baseline_rules.py`:
```
if solar > load: charge battery with surplus
elif price is high and soc > 30%: discharge battery
elif price is low and soc < 50%: charge from grid
else: grid
```

---

## 10. Evaluation

Compare **Grid-only**, **Rule-based** and **RL** on the same held-out days.

| Metric | Formula |
|---|---|
| Energy cost (₹/day) | Σ grid_import × price |
| Renewable utilization (%) | solar used ÷ solar generated |
| Wasted solar (kWh) | curtailed energy |
| Grid dependency (%) | grid import ÷ total load |
| Battery cycles | Σ throughput ÷ (2 × capacity) |
| Forecast error | MAE, MAPE |
| Reliability | unmet load events |

Present results as bar charts and one full-day time-series (solar, load, SoC, price, chosen actions).

---

## 11. Dashboard

Grafana panels: live power flow (solar / load / grid / battery), SoC gauge, price vs. time, forecast vs. actual, agent decisions log, cost savings today/this month, alarms (grid OFF, high temperature).

---

## 12. Deployment (Docker Compose)

Services: `mosquitto`, `timescaledb`, `ingestion`, `forecaster`, `coordinator`, `api`, `grafana`. One command: `docker compose up -d`.

---

## 13. Implementation Plan

| Week | Work | Output |
|---|---|---|
| 1 | Wiring, Modbus map, ESP32 sensors, MQTT | Live data in DB |
| 2 | Cleaning, feature pipeline, Grafana live view | Clean dataset + dashboard |
| 3 | Demand and solar forecasting | Trained models, MAE report |
| 4 | Agents + rule-based coordinator | Working baseline |
| 5 | RL environment, PPO training on logs/simulator | Trained policy |
| 6 | Safety layer, evaluation, comparison charts | Results |
| 7 | Shadow mode (RL suggests, does not control) | Validation |
| 8 | Closed-loop control, final report and demo | Deliverable |

**Tip:** Data collection takes time. Start `generate_data.py` (synthetic solar curve + load pattern + ToD tariff) immediately so weeks 3 to 6 are not blocked.

---

## 14. Risks and Limitations

- Modbus register maps differ per device; verify with the datasheet and a Modbus test tool before coding.
- RL trained only in simulation may not transfer; use shadow mode first.
- Little historical data means weak forecasts; start with LightGBM + weather API features.
- Always keep hardware protections (BMS limits, breakers) independent of the software.

---

## 15. Conclusion

MAEMS converts raw device data into meaningful values, forecasts demand and solar output, lets four specialized agents summarize the situation and uses a PPO-based coordinator (with a safety layer and a rule-based fallback) to decide when to use solar, battery or grid. The result is lower cost, less wasted solar and higher renewable utilization with reliable supply.
