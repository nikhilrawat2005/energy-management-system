# 📘 MAEMS Complete Study & Reference Guide (Master Documentation)

> **Document Purpose:** Yeh document is project ki complete technical aur conceptual study guide hai. Isme system ki working, har ek input-output, internal architecture, AI models, hardware protection, aur Grafana screens ka aasan bhasha mein detailed explanation hai taaki aap isse kabhi bhi study kar sakein ya viva/interviews mein confidently explain kar sakein.

---

## 📑 Table of Contents
1. [Executive Summary (Project Asal Mein Kya Hai?)](#1-executive-summary)
2. [Core Problem & The "Energy Trilemma"](#2-core-problem--the-energy-trilemma)
3. [System Architecture & End-to-End Flow](#3-system-architecture--end-to-end-flow)
4. [The 5 Autonomous Agents (Deep Dive)](#4-the-5-autonomous-agents-deep-dive)
5. [Decision Engine: Reinforcement Learning & Rules](#5-decision-engine-reinforcement-learning--rules)
6. [Hardware Safety & Interlock Guard Layer](#6-hardware-safety--interlock-guard-layer)
7. [Complete Input & Output Specifications](#7-complete-input--output-specifications)
8. [Platforms & Technology Stack Used](#8-platforms--technology-stack-used)
9. [The 4 Grafana Dashboards Explained (Screen-by-Screen)](#9-the-4-grafana-dashboards-explained)
10. [Automated Test Suite & Quality Assurance](#10-automated-test-suite--quality-assurance)
11. [Step-by-Step Viva & Presentation Walkthrough](#11-step-by-step-viva--presentation-walkthrough)

---

## 1. Executive Summary

**MAEMS (Multi-Agent Microgrid Energy Management System)** ek intelligent, autonomous energy controller hai jo kisi building, college campus, ya industrial microgrid mein bijli ke flow ko optimize karta hai.

Microgrid mein aam taur par 3 sources hote hain:
1. **Rooftop Solar PV Array:** Din mein muft (free) green energy generate karta hai.
2. **Battery Energy Storage System (BESS):** Surplus energy ko store karta hai taaki zaroorat ke waqt use kiya ja sake.
3. **Utility Power Grid (Sarkar ki Bijli):** Main power backup, lekin iska price (tariff) din ke alag-alag samay par badalta rehta hai (Time-of-Day Tariff).

**Is Project Ka Main Objective:**
> *"Sarkar ke mehnge tariff hours mein utility grid se bijli khareedna zero karna, solar generation ka 100% self-consumption karna, battery ki physical health aur thermal safety ko protect karna, aur net electricity bill (in ₹) ko drastically kam karna."*

---

## 2. Core Problem & The "Energy Trilemma"

Bina kisi intelligent system ke, aam microgrids mein yeh 3 problems aati hain:
1. **Solar Wastage (Curtailment):** Dopehar ko dhoop sabse zyada hoti hai par factory mein agar load kam hai toh extra solar waste ho jata hai.
2. **Expensive Peak-Hour Buying:** Shaam ko (6 PM - 10 PM) factory ka load peak par hota hai, suraj dhal chuka hota hai, aur bijli board sabse mehnga rate (₹11.5 / kWh) charge karta hai.
3. **Battery Degradation & Hazards:** Agar battery ko bina temperature dekhe ya 100% / 0% par continuously cycle kiya jaye toh cells permanently damage ho jaate hain aur thermal runaway (aag lagne) ka risk hota hai.

**MAEMS ka Solution:**
System aane wale 1 ghante aur 4 ghante ki dhoop (solar) aur load demand ko **Machine Learning** se predict karta hai, 4 specialized agents aapas mein consensus banate hain, aur AI controller battery ko optimal time par charge/discharge karta hai.

---

## 3. System Architecture & End-to-End Flow

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      STEP 1: SENSORS & INGESTION                       │
 │  - Modbus RTU/TCP IEEE-754 Smart Meters (Solar kW, Load kW, Grid V/A)  │
 │  - Weather Pyranometer (W/m2) + Open-Meteo Forecast REST API           │
 │  - Time-of-Day Utility Tariff Feed (₹/kWh)                             │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
 ┌──────────────────────────────────▼─────────────────────────────────────┐
 │                STEP 2: PREDICTIVE FORECASTING MODELS                   │
 │  - Gradient Boosting Regressor: 1h & 4h Solar Irradiance Forecast     │
 │  - Gradient Boosting Regressor: 1h & 4h Site Load Demand Forecast     │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
 ┌──────────────────────────────────▼─────────────────────────────────────┐
 │                   STEP 3: MULTI-AGENT CONSENSUS                        │
 │  [SolarAgent]     -> Calculates current surplus & cloud-drop alarm     │
 │  [DemandAgent]    -> Checks peak-penalty risk & consumption spikes     │
 │  [BatteryAgent]   -> Checks SoC headroom (kW) & thermal derating       │
 │  [PriceAgent]     -> Maps tariff into OFF_PEAK, NORMAL, PEAK tiers     │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
 ┌──────────────────────────────────▼─────────────────────────────────────┐
 │                 STEP 4: ACTION SELECTION (COORDINATOR)                 │
 │  - Q-Learning RL Policy (Trained for cost minimization)                │
 │  - Deterministic Rule-Based Fallback Engine                            │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
 ┌──────────────────────────────────▼─────────────────────────────────────┐
 │                 STEP 5: DETERMINISTIC SAFETY INTERLOCK                 │
 │  - Prevents Overcharge (SoC > 95%) & Overdischarge (SoC < 10%)         │
 │  - Critical Thermal Cutoff (> 55°C) & Grid Blackout Islanding Mode     │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         │                                                     │
 ┌───────▼────────────────────────────┐      ┌─────────────────▼──────────┐
 │ STEP 6A: LOCAL OLLAMA LLM          │      │ STEP 6B: PERSISTENCE & UI  │
 │ - Model: qwen2.5:1.5b via Docker   │      │ - TimescaleDB (Port 5434)  │
 │ - Generates natural language "WHY" │      │ - Mosquitto MQTT (1883)    │
 │   audit narrative & 4h outlook     │      │ - 4-Tier Grafana (3001)    │
 └────────────────────────────────────┘      └────────────────────────────┘
```

---

## 4. The 5 Autonomous Agents (Deep Dive)

Har agent ka apna specific domain aur mathematically isolated responsibility hai:

### 1. `SolarAgent` ([`agents/solar_agent.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/agents/solar_agent.py))
* **Role:** Solar PV Generation Specialist.
* **Inputs:** Instantaneous Solar AC kW, Inverter DC bus Voltage & Current, Pyranometer Irradiance ($W/m^2$), 1h/4h solar forecast.
* **Key Logic:**
  - Surplus calculate karta hai: $\text{Surplus} = \max(0, \text{Solar} - \text{Load})$.
  - Cloud Drop Alarm: Agar pyranometer dhoop detect kar raha hai par inverter se generation achanak gir gayi, toh fault trigger karta hai.

### 2. `DemandAgent` ([`agents/demand_agent.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/agents/demand_agent.py))
* **Role:** Facility Load & Consumption Specialist.
* **Inputs:** Smart Meter Active Load (kW), Power Factor (PF), Line Current (A), 1h/4h demand forecast.
* **Key Logic:**
  - Peak Risk Detection: Agar site load transformer sanction limit (6.0 kW) cross karta hai toh bijli board heavy penalty lagata hai. Yeh agent coordinator ko battery discharge karke peak-shave karne ka vote deta hai.
  - Spike Delta Check: Instantaneous load agar 1-hour forecast se $\ge 2.0\text{ kW}$ zyada ho jaye toh spike alarm deta hai.

### 3. `BatteryAgent` ([`agents/battery_agent.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/agents/battery_agent.py))
* **Role:** Electrochemical & BMS Safety Specialist.
* **Inputs:** State of Charge fraction ($0.0 - 1.0$), NTC pack temperature ($^\circ\text{C}$), Terminal Voltage, Battery current.
* **Key Logic:**
  - Dynamic Headroom: Battery kitne kW absorb ya discharge kar sakti hai bina boundary hit kiye:
    $$\text{Charge Limit (kW)} = \frac{(\text{SoC}_{\max} - \text{SoC}) \times \text{Capacity}}{\Delta t}$$
  - Thermal Throttling:
    - $< 45^\circ\text{C}$: Normal healthy state.
    - $45^\circ\text{C} - 55^\circ\text{C}$: High temperature — charging current ko 50% derate (cut) kar deta hai.
    - $\ge 55^\circ\text{C}$: Critical overheat — battery ko unconditionally isolate karta hai.

### 4. `PriceAgent` ([`agents/price_agent.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/agents/price_agent.py))
* **Role:** Economic Market Specialist.
* **Inputs:** Live Tariff rate (₹/kWh), Current hour of day, Grid stability status ($1 = \text{OK}, 0 = \text{Blackout}$).
* **Key Logic:**
  - Rates ko 3 buckets mein classify karta hai:
    - `OFF_PEAK_CHEAP` (₹4.5 / kWh): Raat 11 PM se subah 6 AM.
    - `NORMAL` (₹7.0 / kWh): Subah 6 AM se shaam 6 PM.
    - `PEAK_EXPENSIVE` (₹11.5 / kWh): Shaam 6 PM se raat 10 PM.
  - Coordinator ko cheap window mein battery grid se bhari rakhne aur peak window mein grid se 0 draw karne ka signal deta hai.

### 5. `AIOrchestratorAgent` ([`agents/ai_coordinator_agent.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/agents/ai_coordinator_agent.py))
* **Role:** Central LLM Systems Analyst (Ollama `qwen2.5:1.5b`).
* **Inputs:** Upar ke chaaron agents ke structured outputs + Coordinator ka action + Hardware safety status.
* **Key Logic:**
  - Local containerized LLM ko streaming socket call karta hai aur human-readable paragraph likhta hai explaining:
    1. *Yeh action kyu liya gaya?*
    2. *Agle 4 ghante ka kya forecast hai?*
    3. *Kya koi anomaly/risk detected hai?*
    4. *Actionable strategic recommendation.*

---

## 5. Decision Engine: Reinforcement Learning & Rules

Coordinator ke paas 2 decision modes hain:
1. **Deterministic Rules (`BaselineRuleCoordinator`):**
   - Agar Solar Surplus $> 0$ aur Battery $< 95\%$ $\rightarrow$ Action 1 (`CHARGE_SOLAR`).
   - Agar Peak Tariff chal raha hai aur Battery $> 20\%$ $\rightarrow$ Action 2 (`DISCHARGE_BATT`).
   - Agar Off-Peak Tariff chal raha hai aur Battery $< 50\%$ $\rightarrow$ Action 3 (`CHARGE_GRID`).
   - Otherwise $\rightarrow$ Action 0 (`SOLAR_FIRST`).

2. **Reinforcement Learning Policy (`QLearningEnergyPolicy`):**
   - 9-dimensional state space: `(solar, load, soc, batt_temp, price, grid_status, hour, solar_fc_1h, load_fc_1h)`.
   - Reward function train kiya gaya hai:
     $$\text{Reward} = -(\text{Grid Cost ₹}) + \lambda \times (\text{Solar Self-Consumption}) - \mu \times (\text{Battery Degradation Penalty})$$
   - Policy file: [`models/rl_policy.json`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/models/rl_policy.json).

---

## 6. Hardware Safety & Interlock Guard Layer

AI ya RL model chahe koi bhi command de, command pehle **`coordinator/safety.py`** se pass hoti hai. Safety layer ke paas **veto power** hai:

| Condition | Safety Rule | Action Taken | Status Logged |
|---|---|---|---|
| Battery Temp $\ge 55^\circ\text{C}$ | Thermal Runaway Lockout | Disconnects Battery (Action $\rightarrow 0$) | `CRITICAL_OVERHEAT_LOCKOUT` |
| SoC $\ge 95\%$ aur command charging ki hai | Overcharge Protection | Blocks Charge (Action $\rightarrow 0$) | `SAFETY_OVERRIDE_BATTERY_FULL` |
| SoC $\le 10\%$ aur command discharge ki hai | Deep Discharge Protection | Blocks Discharge (Action $\rightarrow 0$) | `SAFETY_OVERRIDE_BATTERY_DEPLETED` |
| Grid Status $= 0$ (Grid Fail) | Anti-Islanding Protection | Blocks Grid Charging (Action $\rightarrow 0$) | `SAFETY_OVERRIDE_ISLAND_MODE` |
| Normal State | Within Safe Limits | Passes Original Action | `SAFE_CONFIRMED` |

---

## 7. Complete Input & Output Specifications

### 📥 System Inputs (Telemetry Contract)
1. **Solar Array:** `solar_kw` (Power), `solar_voltage_v`, `solar_current_a`, `solar_irradiance` ($W/m^2$), `solar_yield_kwh`.
2. **Site Load:** `meter_power_kw`, `meter_energy_kwh`, `meter_pf` (Power factor), `ct_current_a`, `pt_voltage_v`.
3. **Battery Pack:** `soc` ($0-1$), `soc_pct` ($0-100\%$), `bms_voltage_v`, `bms_current_a`, `bms_power_kw`, `batt_temp_c`.
4. **Grid & Environment:** `price` (₹/kWh), `grid_status` (1/0), `grid_freq_hz`, `ambient_temp_c`, `humidity_pct`, `wind_speed_ms`.

### 📤 System Outputs (Control Actions & Economics)
* **Action Codes:**
  - `0`: `SOLAR_FIRST` (Solar powers load, grid covers balance, battery floats).
  - `1`: `CHARGE_SOLAR` (Solar powers load + excess surplus charges battery).
  - `2`: `DISCHARGE_BATT` (Battery powers load to minimize expensive grid import).
  - `3`: `CHARGE_GRID` (Grid charges battery during cheap off-peak night hours).
* **Economic Metrics:**
  - `grid_only_cost` = Load $\times$ Price $\times \Delta t$
  - `maems_cost` = Grid Import $\times$ Price $\times \Delta t$
  - `total_savings_inr` = $\sum (\text{grid\_only\_cost} - \text{maems\_cost})$ (Cumulative savings in ₹)
* **AI Audit Output:** Detailed English reasoning narrative logged into TimescaleDB `ai_reasoning_log`.

---

## 8. Platforms & Technology Stack Used

| Platform / Service | Host / Port | Kyu Use Hua? (Technical Rationale) |
|---|---|---|
| **Python 3.10+** | Local runtime | Agents execution, RL Q-table inference, and Live Data Streaming loop. |
| **FastAPI + Uvicorn** | `http://127.0.0.1:8000` | REST API for status query, control override, and Swagger interactive documentation. |
| **TimescaleDB** | `localhost:5434` | PostgreSQL extension optimized for fast time-series ingestion using hypertables (`sensor_readings`, `agent_decisions`, `ai_reasoning_log`). |
| **Eclipse Mosquitto** | `localhost:1883` | Industrial MQTT pub/sub broker for IoT sensor data messaging. |
| **Ollama LLM Engine** | `http://localhost:11434` | Local containerized AI inference engine running `qwen2.5:1.5b` with zero cloud cost or latency risk. |
| **Grafana OSS 11.x** | `http://localhost:3001` | Professional dark-theme SCADA/Operations dashboard suite with live auto-refresh. |

---

## 9. The 4 Grafana Dashboards Explained

Har screen ek specific stakeholder ke liye design ki gayi hai:

### Screen 1: Executive Microgrid Overview & Economic Savings
* **Target Audience:** College Director / Plant Owner.
* **Content:**
  - Instantaneous Solar kW, Load kW, Battery SoC %, and Current Electricity Tariff.
  - **Cumulative Cost Savings (₹):** Live running total of money saved compared to having no smart battery system.
  - **Live Power Balance Waveform:** Solar Generation vs Site Load vs Grid Net Flow.

### Screen 2: Weather & Forecasting Intelligence
* **Target Audience:** Solar & Meteorological Engineer.
* **Content:**
  - Pyranometer Irradiance ($W/m^2$), Ambient Temperature, Humidity, and Cloud Cover %.
  - **Machine Learning Forecast Graphs:** 1-hour and 4-hour forward solar prediction curves.

### Screen 3: Battery & Grid Diagnostics
* **Target Audience:** Electrical & Hardware Maintenance Technician.
* **Content:**
  - Battery pack terminal voltage ($48\text{V} - 54\text{V}$), charging/discharging current (A), pack temperature ($^\circ\text{C}$).
  - Grid voltage stability (230V $\pm 10\%$) and frequency (50 Hz $\pm 1\%$).

### Screen 4: Multi-Agent Decisions & Safety Audit
* **Target Audience:** AI Researcher, Evaluator & Viva Examiner.
* **Content:**
  - Individual agent status badges (`SolarAgent`, `DemandAgent`, `BatteryAgent`, `PriceAgent`).
  - **Latest AI Strategic Reasoning Card:** Live natural language output generated by Ollama LLM.
  - **AI Latency & Engine Gauge:** Model response time and active engine source (`OLLAMA`).
  - **Live Telemetry Dynamic Waveform & Action Frequency Donut Chart.**
  - **Safety Interlock Audit Table:** Every hardware confirmation or safety override record.

---

## 10. Automated Test Suite & Quality Assurance

Project mein 12 automated unit tests hain jo prove karte hain ki code robust aur production-grade hai:

### Run Command:
```powershell
python -m unittest discover tests
```

### What The Tests Verify:
1. **`test_battery_thermal_critical_lockout`**: Checks that $T \ge 55^\circ\text{C}$ disconnects battery immediately.
2. **`test_battery_overcharge_prevention`**: Checks that charging is blocked when $\text{SoC} \ge 95\%$.
3. **`test_battery_overdischarge_prevention`**: Checks that discharging is blocked when $\text{SoC} \le 10\%$.
4. **`test_grid_blackout_island_mode`**: Checks microgrid island mode when utility grid fails.
5. **`test_solar_agent_surplus` & `test_demand_agent_peak_risk`**: Validates mathematical agent calculations.
6. **`test_price_agent_peak_tier` & `offpeak_tier`**: Verifies tariff tier classification.
7. **`test_ml_solar_model_inference` & `demand_model`**: Verifies Gradient Boosting forecast outputs.
8. **`test_ai_fallback_reasoning_generation` & `test_ai_coordinator_entrypoint_schema`**: Validates Ollama LLM integration contract.

---

## 11. Step-by-Step Viva & Presentation Walkthrough

Agar viva ya demo mein explain karna ho, toh is sequence mein bolna:

1. **Opening Hook (Problem Statement):**
   > *"Sir, traditional microgrids mein dopehar ka free solar waste ho jata hai aur shaam ko jab tariff sabse mehnga hota hai (₹11.5/kWh), tab building grid se mehngi bijli leti hai. Humne MAEMS banaya hai jo is problem ko solve karta hai."*

2. **The Agents & Decision Engine:**
   > *"Isme 4 specialized domain agents hain: SolarAgent dhoop aur surplus dekhta hai, DemandAgent peak load risk track karta hai, BatteryAgent thermal safety aur health protect karta hai, aur PriceAgent saste-mehnge hours monitor karta hai. Inka consensus ek Reinforcement Learning (Q-learning) model ko jata hai jo optimal battery dispatch select karta hai."*

3. **Hardware Safety:**
   > *"Har decision ek deterministic Safety Interlock Layer se pass hota hai. Agar battery temperature 55°C touch karega ya SoC 95% ho jayegi, toh safety layer AI command ko turant override kar degi taaki battery blast na ho."*

4. **Local LLM & Explainability:**
   > *"Har decision ke peeche ka technical reason humne local Ollama LLM (Qwen-2.5) ke through explain karwaya hai jo bina kisi internet dependency ke private microgrid controller par chalta hai aur live Grafana dashboard par human-readable audit trail deta hai."*

5. **Closing Result:**
   > *"Result yeh hai ki system solar self-consumption maximize karta hai, battery degradation protect karta hai, aur net electricity bill mein significant ₹ savings deliver karta hai."*
