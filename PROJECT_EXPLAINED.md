# 🎯 MAEMS: Simple System Architecture, Predictions & Project Guide

Agar aapko lag raha hai ki project complex ho gaya hai aur samajh nahi aa raha ki:
1. **Asal mein prediction kya ho rahi hai?**
2. **Kyu itne platforms (Ollama, TimescaleDB, Grafana, FastAPI) use huye hain?**
3. **Data kahan se aa raha hai aur output kya nikal raha hai?**
4. **Test cases kaise run hote hain aur unka kya role hai?**

Toh yeh document aapko simple aur crystal clear language mein explain karega!

---

## 1. Asal Problem & Prediction Kya Hai? (In Simple Words)

Socho aapke college ya society ke paas ek **Solar Plant**, ek **Badi Battery**, aur **Main Power Grid** hai.

Har 15 minute mein 2 cheezein change hoti hain:
1. **Dhoop (Solar Power):** Din mein aati hai, raat ko zero, badal aane par gir jaati hai.
2. **Electricity Tariff (Bijli ka rate):** Sarkar/Grid subah aur dopehar normal rate (₹7) leti hai, shaam ko peak hours mein rate ₹11-12 ho jata hai.

### 🔮 Isme Prediction Kya Hoti Hai?
Project mein **Machine Learning Forecasting Models** (`models/solar_forecast.json` aur `models/demand_forecast.json`) hain jo predict karte hain:
- **1 Ghante aur 4 Ghante aage Solar Kitna banega?**
- **1 Ghante aur 4 Ghante aage Factory/Building ki Electricity Demand kitni hogi?**

### 🧠 Decision Kya Nikalta Hai? (Goal)
Prediction dekh kar **Multi-Agent Coordinator** faisla leta hai:
- *"Shaam ko bijli ₹11 hone wali hai aur solar zero ho jayega! Abhi dopehar mein battery ko free solar se charge kar lo, aur shaam ko grid se ek unit bhi mat khareedo — battery se load chalao!"*
👉 **Output:** Minimum Electricity Bill (Net Savings in ₹) + Zero Power Cut!

---

## 2. Platforms & Tech Stack (Har Ek Ka Ek Simple Role)

| Platform | Asan Bhasha Mein Role | Kyu Use Kiya? |
|---|---|---|
| **Python** | System ka Dimaag | Agents, ML Forecasting Models aur Control Loop chalane ke liye. |
| **FastAPI (`8000`)** | Remote Control (API) | Agar mobile app ya web UI ko current status ya manual control chahiye ho. |
| **TimescaleDB (`5434`)** | Digital Register / Logbook | Har second ka Solar, Battery %, Voltage aur AI Decisions store karne ke liye. |
| **Mosquitto MQTT (`1883`)** | Sensor Wire / Radio | IoT hardware sensors (jaise Pyranometer, Smart Meter) se data lene ke liye. |
| **Grafana (`3001`)** | Control Room ki Screen | Executive ko graphs, gauges aur visual waveforms dikhane ke liye. |
| **Ollama (`11434`)** | AI Analyst (LLM) | Graph dekh kar English/Hindi mein human analysis dene ke liye ki *"Yeh decision kyu liya aur aage kya risk hai"*. |

---

## 3. Data Flow: Input Se Output Tak

```text
[Live Weather / Solar Sensor] + [Building Smart Meter]
                       │
                       ▼
      [Machine Learning Predictors (1h & 4h forecast)]
                       │
                       ▼
          [4 Autonomous Agents Consensus]
     - SolarAgent   : Dhoop aur surplus batata hai
     - DemandAgent  : Load peak risk check karta hai
     - BatteryAgent : Battery health aur % check karta hai
     - PriceAgent   : Sasti/Mehngi bijli ka time batata hai
                       │
                       ▼
      [Safety Interlock Guard (Thermal/Overcharge block)]
                       │
                       ▼
      [Final Dispatched Action (0, 1, 2, or 3)]
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[Ollama LLM Reasoning]      [TimescaleDB Database]
  (Explains "WHY")                   │
                                     ▼
                          [Grafana Live Dashboards]
```

---

## 4. Test Cases & Validation (Quality Assurance)

Project ke andar automated test cases hain jo verify karte hain ki koi hardware damage na ho:

### 🛡️ Test Cases Jo Bane Hain (`tests/` directory):
1. **`test_battery_thermal_critical_lockout`**: Battery ka temperature 55°C se upar hone par battery disconnect hoti hai ya nahi?
2. **`test_battery_overcharge_prevention`**: Battery 95% hone par charging rukti hai ya nahi?
3. **`test_battery_overdischarge_prevention`**: Battery 10% se neeche discharge hone se bachti hai ya nahi?
4. **`test_grid_blackout_island_mode`**: Grid fail hone par microgrid safe island mode mein jata hai ya nahi?
5. **`test_solar_agent_surplus` & `test_demand_agent_peak_risk`**: Agents sahi calculation kar rahe hain ya nahi?
6. **`test_ml_solar_model_inference`**: Solar prediction model sahi numbers output kar raha hai ya nahi?
7. **`test_ai_agent.py`**: Ollama reasoning engine aur fallback structure standard format mein reply kar raha hai ya nahi?

### 🧪 Test Cases Run Karne Ki Single Command:
```powershell
python -m unittest discover tests
```
*Yeh command chalte hi saare test cases execute hote hain aur `OK` status show hota hai.*

---

## 5. Summary For Project Viva / Presentation
Agar koi puche: *"Yeh project kya karta hai?"*

> *"Yeh ek Smart Microgrid Energy Management System hai. Yeh solar generation aur factory demand ko predict karta hai, 4 specialized agents ke through decide karta hai ki battery kab charge ya discharge karni hai taaki bijli ka bill sabse kam aaye, hardware safety ko strictly protect karta hai, aur local Ollama LLM ke through har decision ka human-readable explanation Grafana dashboard par live display karta hai."*
