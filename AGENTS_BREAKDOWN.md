# 🤖 MAEMS: Multi-Agent Architecture & Specialist Agent Roster

Aapka sawal bilkul valid hai: **"Kya yeh sach mein Multi-Agent system hai?"**
👉 **Haan! Yeh 100% true Multi-Agent System (MAS) hai.** Isme koi single generic script nahi chal rahi, balki **5 distinct autonomous agents** hain. Har agent ka apna specific domain, mathematical model, responsibility aur voting rights hain.

Neeche har agent ka **Naam, Role, Input Data, Decision Logic, aur Output** alag-alag bataya gaya hai:

---

## 👥 The 5 Autonomous Agents Roster

```text
                                 ┌─────────────────────────────────┐
                                 │      5. AIOrchestratorAgent     │
                                 │   (Central LLM Intelligence)    │
                                 │  - Powered by Ollama Qwen-2.5   │
                                 │  - Explains "WHY" & predicts 4h │
                                 └───────────────▲─────────────────┘
                                                 │
                                 ┌───────────────┴─────────────────┐
                                 │     MultiAgentCoordinator       │
                                 │  (RL Q-Policy + Safety Engine)  │
                                 └───────────────▲─────────────────┘
                                                 │
               ┌──────────────────┬──────────────┴─────┬──────────────────┐
               │                  │                    │                  │
    ┌──────────▼─────────┐ ┌──────▼───────────┐ ┌──────▼───────────┐ ┌────▼─────────────┐
    │   1. SolarAgent    │ │  2. DemandAgent  │ │ 3. BatteryAgent  │ │   4. PriceAgent   │
    │  (Photovoltaic PV) │ │  (Load Metering) │ │ (Electro-Chem)   │ │  (ToD Grid Tariff)│
    └────────────────────┘ └──────────────────┘ └──────────────────┘ └───────────────────┘
```

---

### ☀️ Agent 1: `SolarAgent` (The Photovoltaic Specialist)
* **Code Location:** [`agents/solar_agent.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/agents/solar_agent.py)
* **Real-world Role:** Solar plant ka virtual engineer.
* **Input Data:**
  - Inverter AC power output (`solar_kw`)
  - Pyranometer solar irradiance (`irradiance_wm2`)
  - DC Inverter Voltage & Current
  - Machine learning 1h & 4h solar generation forecast
* **Specialist Decision / Work:**
  - Calculates instantaneous **Solar Surplus** = $\max(0, \text{Solar} - \text{Load})$.
  - Detects **Cloud Drop Alarms**: Agar pyranometer par dhoop hai par panels se bijli nahi aa rahi (fault ya sudden cloud cover).
  - Categorizes status: `peak_production`, `moderate`, ya `night_zero`.
* **Output to Coordinator:**
  ```json
  {
    "agent": "SolarAgent",
    "solar_now_kw": 6.2,
    "current_surplus_kw": 3.1,
    "status": "peak_production",
    "cloud_drop_alarm": false
  }
  ```

---

### 🔌 Agent 2: `DemandAgent` (The Load & Consumption Specialist)
* **Code Location:** [`agents/demand_agent.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/agents/demand_agent.py)
* **Real-world Role:** Factory / Facility ka consumption manager.
* **Input Data:**
  - Smart Energy Meter active load (`load_kw`)
  - Power Factor (PF) & Current transformer line current (`ct_current_a`)
  - ML demand forecast for next 1h and 4h
* **Specialist Decision / Work:**
  - Evaluates **Peak Demand Penalty Risk**: Agar load limit (6.0 kW) cross karta hai toh bijli board heavy penalty lagata hai. Yeh agent pehle hi alert raise karta hai.
  - Detects **Load Spikes**: Jab instantaneous consumption 1-hour forecast se achanak bohot zyada nikal jaye.
* **Output to Coordinator:**
  ```json
  {
    "agent": "DemandAgent",
    "load_now_kw": 3.1,
    "load_category": "moderate",
    "peak_risk": false,
    "spike_detected": false
  }
  ```

---

### 🔋 Agent 3: `BatteryAgent` (The Electrochemical & BMS Specialist)
* **Code Location:** [`agents/battery_agent.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/agents/battery_agent.py)
* **Real-world Role:** Battery pack aur BMS (Battery Management System) ka protector.
* **Input Data:**
  - State of Charge (`soc` fraction 0.0 - 1.0)
  - Pack temperature via NTC thermistor (`batt_temp_c`)
  - Terminal voltage and charge/discharge current
* **Specialist Decision / Work:**
  - Calculates dynamic **Charge & Discharge Headroom (kW)** so cells don't degrade.
  - **Thermal Derating & Safety Flags**:
    - Normal ( $< 45^\circ\text{C}$): `HEALTHY` (Full 5kW C-rate)
    - Hot ( $45^\circ\text{C} - 55^\circ\text{C}$): `THERMAL_THROTTLED` (C-rate 50% cut)
    - Overheat ( $> 55^\circ\text{C}$): `CRITICAL_OVERHEAT_LOCKOUT` (Complete isolation)
* **Output to Coordinator:**
  ```json
  {
    "agent": "BatteryAgent",
    "soc_pct": 65.0,
    "health_flag": "HEALTHY",
    "avail_charge_kw": 3.0,
    "avail_discharge_kw": 3.0,
    "thermal_throttled": false
  }
  ```

---

### 💰 Agent 4: `PriceAgent` (The Economic Market Specialist)
* **Code Location:** [`agents/price_agent.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/agents/price_agent.py)
* **Real-world Role:** Energy market trader jo electricity rates track karta hai.
* **Input Data:**
  - Current utility tariff (`price` in ₹/kWh)
  - Current hour of day (Time-of-Day tariff schedule)
  - Grid availability status (`1 = online`, `0 = blackout`)
* **Specialist Decision / Work:**
  - Maps live rate into tiers: `OFF_PEAK_CHEAP` (₹4.5), `NORMAL` (₹7.0), `PEAK_EXPENSIVE` (₹11.5).
  - Flags economic windows: Cheap hours mein battery charge karne ka signal deta hai, peak hours mein grid se connection minimize karne ka vote deta hai.
* **Output to Coordinator:**
  ```json
  {
    "agent": "PriceAgent",
    "price_inr_per_kwh": 11.5,
    "price_tier": "PEAK_EXPENSIVE",
    "cheap_window": false,
    "peak_window": true,
    "grid_ok": true
  }
  ```

---

### 🧠 Agent 5: `AIOrchestratorAgent` (The Central LLM Intelligence)
* **Code Location:** [`agents/ai_coordinator_agent.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/agents/ai_coordinator_agent.py)
* **Real-world Role:** Chief Systems Analyst (Ollama LLM).
* **Input Data:**
  - Sabhi 4 agents (`Solar`, `Demand`, `Battery`, `Price`) ke reports.
  - Final dispatched action from RL/Rules.
  - Hardware safety confirmations.
* **Specialist Decision / Work:**
  - Multi-agent data ko synthesize karke natural language mein audit trail generate karta hai:
    1. *Action kyu select hua?*
    2. *Agle 4 ghante mein kya hone wala hai?*
    3. *Kya koi risk ya anomaly hai?*
    4. *Strategic Recommendation.*
* **Output:** Stored in database table `ai_reasoning_log` and rendered directly on Grafana Dashboard 4.

---

## 🗳️ How The Multi-Agent Consensus Works (Example)

Maan lo shaam ke 7:00 PM baje hain:
1. `SolarAgent` bolta hai: *"Surplus zero hai, dhoop ja chuki hai."*
2. `DemandAgent` bolta hai: *"Building load 4.2 kW hai (moderate)."*
3. `BatteryAgent` bolta hai: *"Mera SoC 91% hai aur temperature 30°C hai (Healthy, can discharge up to 5 kW)."*
4. `PriceAgent` bolta hai: *"Tariff ₹11.5 PEAK_EXPENSIVE chal raha hai — grid se bijli lena bohot mehnga padega!"*
5. **Coordinator Consensus:** Sabhi agents ka data match hua ➡️ Action dispatched: **`DISCHARGE_BATT`** (Battery powers building, zero grid cost).
6. `AIOrchestratorAgent` (Ollama): Reasoning likhta hai:
   > *"Action DISCHARGE_BATT is optimal because tariff is in peak tier (₹11.5/kWh) and battery SoC is sufficient at 91%. Recommend holding battery discharge until peak tariff concludes at 22:00."*

---

## 📌 Summary for Viva
Agar examiner puche: *"Kaun-kaun se agents hain aur kaise communicate karte hain?"*
> *"MAEMS mein 5 autonomous agents hain: SolarAgent (PV surplus & cloud drop alarms), DemandAgent (peak load risk & consumption spikes), BatteryAgent (SoC headroom & thermal derating), PriceAgent (Time-of-Day economic tariff tiers), aur Central AIOrchestratorAgent (Ollama LLM narrative synthesis). Sabhi agents Coordinator ko structured reports bhejte hain, jiske base par Reinforcement Learning policy optimal action select karti hai aur Safety interlock hardware ko protect karta hai."*
