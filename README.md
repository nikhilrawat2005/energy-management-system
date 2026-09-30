# ⚡ Multi-Agent Energy Management System (MAEMS) with Local Ollama LLM Reasoning

An end-to-end autonomous Microgrid Energy Management System that integrates **IEEE-754 Modbus Sensor Ingestion**, **4 Autonomous Domain Agents**, **Rule-based & Reinforcement Learning (Q-Learning) Dispatch**, **Hardware Safety Interlocks**, **TimescaleDB Time-Series Telemetry**, **Mosquitto MQTT Messaging Broker**, **Ollama Local LLM (qwen2.5:1.5b)** for strategic scenario reasoning, and a **4-Tier Professional Grafana Operations Suite**.

---

## 🌟 Architecture & System Overview

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                    1. Telemetry Ingestion Layer                        │
 │  - Real-world / Data4Cyber Modbus IEEE-754 floating-point register     │
 │  - Open-Meteo REST API live weather stream (Irradiance, Temp, Wind)   │
 │  - Time-of-Day (ToD) Dynamic Tariff Feed                               │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
 ┌──────────────────────────────────▼─────────────────────────────────────┐
 │                2. Autonomous Multi-Agent Consensus Tier                │
 │  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ ┌─────────┐│
 │  │   SolarAgent    │ │   DemandAgent   │ │ BatteryAgent  │ │PriceAgnt││
 │  │  (PV Yield/Surp)│ │(Load Peak Risk) │ │(SoC/C-Rate/deg)│ │(Tariff) ││
 │  └────────┬────────┘ └────────┬────────┘ └───────┬───────┘ └────┬────┘│
 └───────────┼───────────────────┼──────────────────┼──────────────┼─────┘
             └───────────────────┼──────────────────┼──────────────┘
                                 │
 ┌───────────────────────────────▼────────────────────────────────────────┐
 │         3. Core Decision & Hardware Interlock Coordinator              │
 │  - Reinforcement Learning Policy (Q-Learning / Bellman Action Space)   │
 │  - Deterministic Rule-Based Fallback Engine                            │
 │  - Safety Interlock Guard: Overcharge/Overdischarge & Islanding        │
 └───────────────────────────────┬────────────────────────────────────────┘
                                 │
         ┌───────────────────────┴────────────────────────┐
         │                                                │
 ┌───────▼──────────────────────────┐     ┌───────────────▼───────────────┐
 │ 4. Local Ollama LLM Reasoning    │     │ 5. Time-Series Telemetry DB   │
 │ - Model: qwen2.5:1.5b via Docker │     │ - TimescaleDB (PostgreSQL 15) │
 │ - Streaming Socket Architecture  │     │ - MQTT Broker (Mosquitto)     │
 │ - Multi-Agent Synthesis & 4h Fc  │     │ - Hypertables: sensor_readings│
 └────────────────┬─────────────────┘     │   agent_decisions, ai_log     │
                  │                       └───────────────┬───────────────┘
                  │                                       │
 ┌────────────────▼───────────────────────────────────────▼───────────────┐
 │              6. Grafana 4-Tier Operations & Audit Suite                │
 │  Page 1: Executive Microgrid Overview & Economic Savings               │
 │  Page 2: Solar Irradiance, Pyranometer & Weather Intelligence          │
 │  Page 3: Battery Electrochemical Health & Grid Power Diagnostics       │
 │  Page 4: Multi-Agent Decisions, Ollama LLM Reasoning & Safety Audit   │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

1. **Autonomous Multi-Agent Consensus**:
   - `SolarAgent`: Monitors PV irradiance, inverter voltage/current, and calculates real-time solar surplus headroom.
   - `DemandAgent`: Analyzes load demand patterns, power factor (PF), and detects impending peak consumption spikes.
   - `BatteryAgent`: Tracks electrochemical State of Charge (SoC 10%–95%), cell temperatures, and dynamically throttles C-rates.
   - `PriceAgent`: Evaluates Time-of-Day (ToD) tariff windows (`OFF_PEAK`, `NORMAL`, `PEAK_EXPENSIVE`) to trigger cost minimization.

2. **Ollama Local LLM Agent Integration (`qwen2.5:1.5b`)**:
   - Deployed locally via Docker with zero cloud dependencies or API keys.
   - Synthesizes state snapshots and produces concise, human-readable explanations of *why* an action was chosen, risks detected, and a 4-hour strategic forecast.
   - Socket streaming prevents timeouts on CPU-only infrastructure with automatic deterministic heuristic fallbacks.

3. **Deterministic Safety Layer**:
   - Independent hardware interlock preventing overcharge ($SoC > 95\%$), deep discharge ($SoC < 10\%$), and thermal runaway ($T > 45^\circ\text{C}$).
   - Microgrid anti-islanding protection switching when grid voltage/frequency destabilizes.

4. **Industrial Visualization (Grafana 4-Tier Suite)**:
   - **Unified Navigation Navbar** on every page with active route highlighting.
   - Live streaming updates every 2 seconds.
   - Dark engineering aesthetic tailored for control room displays.

---

## 📥 Inputs & Outputs Specification

### Inputs
- **Solar Telemetry**: PV AC power (kW), Inverter DC bus voltage (V), Inverter current (A), Pyranometer irradiance ($W/m^2$).
- **Load Telemetry**: Site active power (kW), line voltage (V), grid frequency (Hz), power factor (PF), load current (A).
- **Battery Pack BMS**: Terminal voltage (V), charging/discharging current (A), pack temperature ($^\circ\text{C}$), State of Charge fraction ($0.0 - 1.0$).
- **Grid & Market**: Time-of-day tariff rates (Rs./kWh), grid status ($1 = \text{online}, 0 = \text{islanded}$).
- **Weather API**: Open-Meteo live solar irradiance, cloud cover %, ambient temperature, and wind speed.

### Outputs
- **Control Actions**:
  - `0: IDLE` (System floating, zero battery cycling)
  - `1: CHARGE_SOLAR` (PV surplus directed to battery pack)
  - `2: DISCHARGE_BATT` (Battery powers facility to shave grid load)
  - `3: CHARGE_GRID` (Grid charges battery during cheap off-peak hours)
- **Economic Metrics**: Real-time avoided utility cost, cumulative savings (Rs.), grid energy balance.
- **Safety Interlock Audits**: State flags (`SAFE_CONFIRMED`, `BATTERY_FULL`, `BATTERY_DEPLETED`, `THERMAL_LOCKOUT`).
- **AI Strategic Narratives**: Natural language audit logs written by Ollama to TimescaleDB.

---

## 🛠️ Technology Stack & Platforms

| Component | Platform / Tech | Description |
|-----------|-----------------|-------------|
| **Core Runtime** | Python 3.10+ / 3.14 | Coordination loop, agent heuristics, and RL policy |
| **API Backend** | FastAPI + Uvicorn | REST API & OpenAPI docs at `http://127.0.0.1:8000/docs` |
| **Time-Series DB**| TimescaleDB (PostgreSQL 15) | Dockerized hypertable database running on port `5434` |
| **MQTT Broker** | Eclipse Mosquitto 2.0 | Sensor publishing and telemetry messaging on port `1883` |
| **AI LLM Engine** | Ollama (`qwen2.5:1.5b`) | Local edge-deployable containerized LLM on port `11434` |
| **Dashboards** | Grafana OSS 11.x | Multi-page visualization running on port `3001` |
| **Weather Feed** | Open-Meteo REST API | Solar irradiance and temperature forecasting |

---

## ⚡ Quick Start & Run Guide

### 1. Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
- Python 3.10+ installed.

### 2. Single-Command Launch
Start the entire infrastructure (TimescaleDB, Grafana, Mosquitto, Ollama, FastAPI, and Real-Time Telemetry Streamer):

```bash
python start.py
```

### 3. Access Services
- **Grafana Dashboards**: [http://localhost:3001](http://localhost:3001) *(User: `admin` / Password: `admin`)*
  - **1. Microgrid Overview**: [http://localhost:3001/d/maems_dashboard_v1/1-maems-executive-microgrid-overview](http://localhost:3001/d/maems_dashboard_v1/1-maems-executive-microgrid-overview)
  - **2. Weather & Forecasts**: [http://localhost:3001/d/maems_weather_v1/2-maems-weather-and-forecasting-intelligence](http://localhost:3001/d/maems_weather_v1/2-maems-weather-and-forecasting-intelligence)
  - **3. Battery Diagnostics**: [http://localhost:3001/d/maems_battery_grid_v1/3-maems-battery-and-grid-diagnostics](http://localhost:3001/d/maems_battery_grid_v1/3-maems-battery-and-grid-diagnostics)
  - **4. Multi-Agent & AI Audit**: [http://localhost:3001/d/maems_agents_v1/4-maems-multi-agent-decisions-and-safety-audit](http://localhost:3001/d/maems_agents_v1/4-maems-multi-agent-decisions-and-safety-audit)
- **FastAPI Interactive Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **TimescaleDB Port**: `localhost:5434`
- **Ollama Engine Port**: `localhost:11434`

---

## 🧪 Evaluation & Benchmarks

Run comparative simulations across 7 days of historical profiles:

```bash
python evaluate.py
```

Compares **Grid-Only Baseline**, **Heuristic Rule-Based Coordinator**, and **Reinforcement Learning Policy** on solar self-consumption, battery degradation, and net utility bill savings.

---

## 🔒 Safety Compliance
All control outputs are validated by `coordinator/safety.py` prior to dispatch. If sensor readings exceed operational safety envelopes defined in `config/limits.yaml`, safety interlocks unconditionally override RL/heuristic commands and log hardware safety alerts.
