"""
MAEMS Streamlit Dashboard
=========================
A self-contained interactive dashboard for the Multi-Agent Energy Management System.
Runs completely without Docker / TimescaleDB / Grafana — uses the core Python engine
directly so it deploys instantly on Streamlit Cloud.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── path bootstrap ──────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from coordinator.coordinator import MultiAgentCoordinator

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ MAEMS — Energy Management",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark engineering palette */
:root {
    --bg-primary: #0d1117;
    --accent-solar: #f6c90e;
    --accent-battery: #3fb950;
    --accent-grid: #58a6ff;
    --accent-load: #f78166;
    --accent-safe: #3fb950;
    --accent-warn: #d29922;
    --accent-danger: #f85149;
}
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 4px 0;
}
.metric-card h3 { margin: 0; font-size: 0.78rem; color: #8b949e; letter-spacing: 0.06em; text-transform: uppercase; }
.metric-card .value { font-size: 2rem; font-weight: 700; color: #e6edf3; line-height: 1.2; }
.metric-card .sub { font-size: 0.75rem; color: #8b949e; margin-top: 2px; }
.action-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
}
.safe   { background:#1a3a2a; color:#3fb950; border:1px solid #238636; }
.warn   { background:#3a2e1a; color:#d29922; border:1px solid #9e6a03; }
.danger { background:#3a1a1a; color:#f85149; border:1px solid #da3633; }
.stButton>button { border-radius:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── cached coordinator ───────────────────────────────────────────────────────
@st.cache_resource
def get_coordinator() -> MultiAgentCoordinator:
    return MultiAgentCoordinator()


# ── simulation helpers ───────────────────────────────────────────────────────
def solar_profile(hour: float) -> float:
    """Realistic bell-curve solar output (peak at 13:00)."""
    if hour < 6 or hour > 19:
        return 0.0
    peak = 8.5
    return max(0.0, peak * math.exp(-0.5 * ((hour - 13) / 3.2) ** 2))


def load_profile(hour: float) -> float:
    """Residential load profile with morning and evening peaks."""
    base = 2.5
    morning = 1.5 * math.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
    evening = 3.0 * math.exp(-0.5 * ((hour - 20) / 2.0) ** 2)
    return base + morning + evening + random.gauss(0, 0.15)


def price_tod(hour: float) -> float:
    """Time-of-day tariff (INR/kWh)."""
    if 6 <= hour < 10 or 18 <= hour < 23:
        return random.uniform(10.5, 13.0)   # peak
    if 23 <= hour or hour < 6:
        return random.uniform(3.5, 5.0)     # off-peak
    return random.uniform(6.5, 8.5)         # normal


def build_state(hour: float, soc: float) -> Dict[str, Any]:
    solar = solar_profile(hour) * random.uniform(0.85, 1.15)
    load  = max(0.5, load_profile(hour))
    irr   = max(0.0, 950 * math.exp(-0.5 * ((hour - 13) / 3.5) ** 2) * random.uniform(0.8, 1.2))
    price = price_tod(hour)
    return {
        "solar_kw":       round(solar, 2),
        "load_kw":        round(load, 2),
        "soc":            round(max(0.1, min(0.95, soc)), 3),
        "batt_temp":      round(27 + random.gauss(0, 1.5), 1),
        "price":          round(price, 2),
        "irradiance_wm2": round(irr, 1),
        "grid_voltage":   round(random.gauss(230, 1.5), 1),
        "grid_freq":      round(random.gauss(50.0, 0.05), 3),
        "grid_status":    1,
        "hour":           hour,
        "solar_fc_1h":    round(solar_profile(hour + 1) * 0.9, 2),
        "load_fc_1h":     round(load_profile(hour + 1), 2),
        "solar_fc_4h":    round(solar_profile(hour + 4) * 0.85, 2),
        "load_fc_4h":     round(load_profile(hour + 4), 2),
    }


def simulate_day(coordinator: MultiAgentCoordinator, mode: str, steps: int = 96) -> pd.DataFrame:
    """Simulate a full 24-hour day in `steps` ticks."""
    rows: List[Dict] = []
    soc = 0.50
    batt_cap_kwh = 20.0
    tick_h = 24.0 / steps

    for i in range(steps):
        hour = i * tick_h
        state = build_state(hour, soc)
        result = coordinator.process_tick(state, mode=mode, current_time_sec=hour * 3600)

        action = result["final_action"]
        solar  = state["solar_kw"]
        load   = state["load_kw"]
        soc_now = state["soc"]

        # Simple SoC dynamics
        if action == 1:   # CHARGE_SOLAR
            soc = min(0.95, soc + (solar - load) * tick_h * 0.9 / batt_cap_kwh)
        elif action == 2:  # DISCHARGE_BATT
            soc = max(0.10, soc - load * 0.5 * tick_h / batt_cap_kwh)
        elif action == 3:  # CHARGE_GRID
            soc = min(0.95, soc + 2.0 * tick_h / batt_cap_kwh)

        grid_import = max(0.0, load - solar) if action != 2 else 0.0
        savings     = grid_import * state["price"] * 0.35 / 100  # approx

        rows.append({
            "hour":         hour,
            "time":         f"{int(hour):02d}:{int((hour % 1)*60):02d}",
            "solar_kw":     solar,
            "load_kw":      load,
            "soc_pct":      soc_now * 100,
            "price":        state["price"],
            "action":       action,
            "action_name":  result["final_action_name"].split("(")[0].strip(),
            "safety_status":result["safety_status"],
            "safety_override": result["safety_override"],
            "grid_import":  round(grid_import, 2),
            "batt_temp":    state["batt_temp"],
            "irradiance":   state["irradiance_wm2"],
            "savings_rs":   round(savings, 2),
        })
    return pd.DataFrame(rows)


# ── colour maps ──────────────────────────────────────────────────────────────
ACTION_COLORS = {
    "SOLAR_FIRST":    "#58a6ff",
    "CHARGE_SOLAR":   "#f6c90e",
    "DISCHARGE_BATT": "#3fb950",
    "CHARGE_GRID":    "#bc8cff",
}
SAFETY_COLORS = {
    "SAFE_CONFIRMED":    "#3fb950",
    "BATTERY_FULL":      "#f6c90e",
    "BATTERY_DEPLETED":  "#f78166",
    "THERMAL_LOCKOUT":   "#f85149",
    "GRID_BLACKOUT":     "#f85149",
}


def action_color(name: str) -> str:
    for k, v in ACTION_COLORS.items():
        if name.startswith(k):
            return v
    return "#8b949e"


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/electrical.png",
        width=60,
    )
    st.sidebar.title("⚡ MAEMS")
    st.sidebar.caption("Multi-Agent Energy Management System")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Live Control", "📊 Day Simulation", "🤖 Agent Inspector", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")
    mode = st.sidebar.selectbox("Dispatch Mode", ["rl", "rule_based"], index=0,
                                help="RL = Q-Learning policy | Rule-Based = deterministic heuristics")
    sim_steps = st.sidebar.slider("Simulation Steps", 48, 288, 96, 48)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Stack:** Python · FastAPI · RL · Streamlit\n\n"
        "**GitHub:** [nikhilrawat2005](https://github.com/nikhilrawat2005/energy-management-system)"
    )
    return page, mode, sim_steps


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — LIVE CONTROL
# ═══════════════════════════════════════════════════════════════════════════
def page_live_control(coord: MultiAgentCoordinator, mode: str):
    st.title("🏠 Live Control Panel")
    st.caption("Adjust telemetry inputs and dispatch through the full agent pipeline in real-time.")
    st.markdown("---")

    # ── Input sliders ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("☀️ Solar & Load")
        solar_kw  = st.slider("Solar Output (kW)",     0.0, 12.0, 5.4,  0.1)
        load_kw   = st.slider("Load Demand (kW)",       0.5, 10.0, 3.1,  0.1)
        irr       = st.slider("Irradiance (W/m²)",       0,  1200,  720,   10)
    with col2:
        st.subheader("🔋 Battery")
        soc       = st.slider("State of Charge (%)",   10,   95,   62,    1) / 100
        batt_temp = st.slider("Battery Temp (°C)",     15.0, 55.0, 31.0,  0.5)
        grid_stat = st.selectbox("Grid Status", ["Online (1)", "Blackout (0)"])
        grid_int  = 1 if grid_stat.startswith("Online") else 0
    with col3:
        st.subheader("⚡ Grid & Tariff")
        price       = st.slider("Tariff (INR/kWh)",   2.0, 15.0,  8.5, 0.5)
        grid_volt   = st.slider("Grid Voltage (V)",   200.0, 250.0, 230.0, 0.5)
        grid_freq   = st.slider("Grid Frequency (Hz)", 48.5,  51.5,  50.0, 0.05)
        hour_val    = st.slider("Hour of Day",           0,    24,   14,    1)

    state = {
        "solar_kw": solar_kw, "load_kw": load_kw,
        "soc": soc, "batt_temp": batt_temp,
        "price": price, "irradiance_wm2": irr,
        "grid_voltage": grid_volt, "grid_freq": grid_freq,
        "grid_status": grid_int, "hour": float(hour_val),
        "solar_fc_1h": solar_profile(hour_val + 1) * 0.9,
        "load_fc_1h":  load_profile(hour_val + 1),
        "solar_fc_4h": solar_profile(hour_val + 4) * 0.85,
        "load_fc_4h":  load_profile(hour_val + 4),
    }

    if st.button("🚀 Dispatch Control Tick", use_container_width=True, type="primary"):
        with st.spinner("Running agent pipeline…"):
            result = coord.process_tick(state, mode=mode, current_time_sec=float(hour_val) * 3600)
        st.session_state["last_result"] = result
        st.session_state["last_state"]  = state

    result = st.session_state.get("last_result")
    state_snap = st.session_state.get("last_state", state)

    if result:
        st.markdown("---")
        st.subheader("📋 Dispatch Result")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            surplus = round(state_snap["solar_kw"] - state_snap["load_kw"], 2)
            st.metric("Solar Surplus", f"{surplus:+.2f} kW", delta=f"Solar {state_snap['solar_kw']}kW")
        with c2:
            st.metric("Battery SoC", f"{state_snap['soc']*100:.1f}%", delta=f"Temp {state_snap['batt_temp']}°C")
        with c3:
            action_lbl = result["final_action_name"].split("(")[0].strip()
            st.metric("Final Action", action_lbl)
        with c4:
            override = "⚠️ YES" if result["safety_override"] else "✅ No"
            st.metric("Safety Override", override, delta=result["safety_status"])

        # Agent reports table
        st.subheader("🤖 Agent Reports")
        agents_tab = st.tabs(["☀️ Solar", "⚡ Demand", "🔋 Battery", "💰 Price"])
        reports = result.get("agent_reports", {})
        for tab, (key, label) in zip(agents_tab, [
            ("solar_agent", "Solar"), ("demand_agent", "Demand"),
            ("battery_agent", "Battery"), ("price_agent", "Price")
        ]):
            with tab:
                rep = reports.get(key, {})
                if rep:
                    df = pd.DataFrame([{"Key": k, "Value": v} for k, v in rep.items()])
                    st.dataframe(df, hide_index=True, use_container_width=True)
                else:
                    st.info("No report data")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DAY SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
def page_day_simulation(coord: MultiAgentCoordinator, mode: str, sim_steps: int):
    st.title("📊 24-Hour Day Simulation")
    st.caption(f"Full day simulation · {sim_steps} ticks · Mode: **{mode}**")

    if st.button("▶️  Run Simulation", use_container_width=True, type="primary"):
        with st.spinner(f"Simulating {sim_steps} ticks…"):
            df = simulate_day(coord, mode, sim_steps)
        st.session_state["sim_df"] = df
        st.success(f"✅ Simulation complete — {len(df)} ticks processed")

    df = st.session_state.get("sim_df")
    if df is None:
        st.info("Click **Run Simulation** to generate a 24-hour profile.")
        return

    # ── KPI row ──────────────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("☀️ Peak Solar",   f"{df['solar_kw'].max():.1f} kW")
    k2.metric("🔋 Min SoC",      f"{df['soc_pct'].min():.1f}%")
    k3.metric("🔋 Max SoC",      f"{df['soc_pct'].max():.1f}%")
    k4.metric("⚡ Grid Imports",  f"{df['grid_import'].sum():.1f} kWh")
    k5.metric("💸 Est. Savings",  f"₹{df['savings_rs'].sum():.1f}")

    # ── Power flow chart ─────────────────────────────────────────────────
    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(x=df["hour"], y=df["solar_kw"],  name="Solar (kW)",
                                   fill="tozeroy", line=dict(color="#f6c90e", width=2)))
    fig_power.add_trace(go.Scatter(x=df["hour"], y=df["load_kw"],   name="Load (kW)",
                                   line=dict(color="#f78166", width=2)))
    fig_power.add_trace(go.Scatter(x=df["hour"], y=df["grid_import"], name="Grid Import (kW)",
                                   line=dict(color="#58a6ff", width=1.5, dash="dash")))
    fig_power.update_layout(title="Power Flow Profile", xaxis_title="Hour",
                             yaxis_title="kW", template="plotly_dark",
                             legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig_power, use_container_width=True)

    col_l, col_r = st.columns(2)

    # ── SoC chart ────────────────────────────────────────────────────────
    with col_l:
        fig_soc = go.Figure()
        fig_soc.add_trace(go.Scatter(x=df["hour"], y=df["soc_pct"], name="SoC (%)",
                                     fill="tozeroy", line=dict(color="#3fb950", width=2)))
        fig_soc.add_hline(y=10, line_dash="dot", line_color="#f85149",
                           annotation_text="Min 10%")
        fig_soc.add_hline(y=95, line_dash="dot", line_color="#f6c90e",
                           annotation_text="Max 95%")
        fig_soc.update_layout(title="Battery State of Charge", xaxis_title="Hour",
                               yaxis_title="SoC (%)", template="plotly_dark",
                               yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_soc, use_container_width=True)

    # ── Action distribution ──────────────────────────────────────────────
    with col_r:
        action_counts = df["action_name"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        fig_pie = px.pie(action_counts, names="Action", values="Count",
                         color="Action",
                         color_discrete_map={k.split("(")[0].strip(): v
                                             for k, v in ACTION_COLORS.items()},
                         title="Action Distribution")
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Tariff & irradiance ──────────────────────────────────────────────
    fig_extra = go.Figure()
    fig_extra.add_trace(go.Scatter(x=df["hour"], y=df["price"],      name="Tariff (INR/kWh)",
                                   line=dict(color="#bc8cff", width=1.8), yaxis="y"))
    fig_extra.add_trace(go.Bar(x=df["hour"], y=df["irradiance"] / 100, name="Irradiance ÷100",
                               marker_color="rgba(246,201,14,0.3)", yaxis="y"))
    fig_extra.update_layout(title="Tariff & Irradiance", xaxis_title="Hour",
                             template="plotly_dark", barmode="overlay",
                             legend=dict(orientation="h"))
    st.plotly_chart(fig_extra, use_container_width=True)

    # ── Safety audit ─────────────────────────────────────────────────────
    st.subheader("🔒 Safety Audit")
    overrides = df[df["safety_override"]].shape[0]
    st.markdown(f"- **Total ticks:** {len(df)}  |  **Safety overrides:** {overrides}")
    safety_counts = df["safety_status"].value_counts()
    st.dataframe(safety_counts.rename("Count"), use_container_width=True)

    # ── Raw data expander ─────────────────────────────────────────────────
    with st.expander("📄 View Raw Simulation Data"):
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "maems_simulation.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — AGENT INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════
def page_agent_inspector(coord: MultiAgentCoordinator):
    st.title("🤖 Agent Inspector")
    st.caption("Send a telemetry snapshot and inspect each agent's structured opinion.")

    with st.form("agent_form"):
        c1, c2 = st.columns(2)
        with c1:
            solar_kw  = st.number_input("Solar (kW)",        0.0,  15.0, 5.4,  0.1)
            load_kw   = st.number_input("Load (kW)",          0.5,  12.0, 3.1,  0.1)
            soc_pct   = st.number_input("SoC (%)",           10,     95,   62,    1)
        with c2:
            price     = st.number_input("Tariff (INR/kWh)",  2.0,  15.0,  8.5,  0.5)
            batt_temp = st.number_input("Batt Temp (°C)",   15.0,  55.0, 31.0,  0.5)
            hour_val  = st.number_input("Hour of Day",         0,    24,   14,    1)
        submitted = st.form_submit_button("🔍 Inspect Agents", use_container_width=True, type="primary")

    if submitted:
        state = {
            "solar_kw": solar_kw, "load_kw": load_kw,
            "soc": soc_pct / 100, "batt_temp": batt_temp,
            "price": price, "irradiance_wm2": 600.0,
            "grid_voltage": 230.0, "grid_freq": 50.0,
            "grid_status": 1, "hour": float(hour_val),
            "solar_fc_1h": solar_kw * 0.9, "load_fc_1h": load_kw * 1.05,
            "solar_fc_4h": solar_kw * 0.7, "load_fc_4h": load_kw * 1.1,
        }

        agent_defs = [
            ("☀️ Solar Agent",   coord.solar_agent),
            ("⚡ Demand Agent",  coord.demand_agent),
            ("🔋 Battery Agent", coord.battery_agent),
            ("💰 Price Agent",   coord.price_agent),
        ]

        cols = st.columns(2)
        for idx, (title, agent) in enumerate(agent_defs):
            rep = agent.report(state)
            with cols[idx % 2]:
                st.subheader(title)
                for k, v in rep.items():
                    if isinstance(v, float):
                        st.metric(k.replace("_", " ").title(), f"{v:.3f}")
                    elif isinstance(v, bool):
                        color = "🟢" if v else "🔴"
                        st.write(f"{color} **{k}**: {v}")
                    else:
                        st.write(f"**{k}:** {v}")
                st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════
def page_about():
    st.title("ℹ️ About MAEMS")
    st.markdown("""
## ⚡ Multi-Agent Energy Management System (MAEMS)

An end-to-end autonomous **Microgrid Energy Management System** featuring:

| Component | Technology |
|---|---|
| **Multi-Agent Core** | 4 specialist agents (Solar, Demand, Battery, Price) |
| **Decision Engine** | Q-Learning RL Policy + Deterministic Rules fallback |
| **Safety Interlock** | Hardware-grade overcharge/thermal/islanding protection |
| **Forecasting** | Custom Gradient Boosting for 1h & 4h ahead solar/demand |
| **AI Reasoning** | Ollama Local LLM (qwen2.5:1.5b) — offline capable |
| **API** | FastAPI REST with OpenAPI/Swagger docs |
| **Live Telemetry** | TimescaleDB + Mosquitto MQTT (Docker stack) |
| **Dashboards** | Grafana 4-tier ops suite + this Streamlit app |

### 🤖 The 4 Specialist Agents

- **☀️ SolarAgent** — Tracks PV irradiance, calculates real-time surplus headroom
- **⚡ DemandAgent** — Analyzes load patterns, detects impending peak spikes
- **🔋 BatteryAgent** — Monitors SoC (10%–95%), cell temperature, C-rate throttling
- **💰 PriceAgent** — Evaluates ToD tariff windows for cost minimization

### 🎯 Control Actions
| ID | Action | Description |
|---|---|---|
| 0 | SOLAR_FIRST | Solar to load, grid covers balance |
| 1 | CHARGE_SOLAR | Surplus PV directed to battery |
| 2 | DISCHARGE_BATT | Battery powers facility (grid shaving) |
| 3 | CHARGE_GRID | Off-peak grid charges battery |

### 🔒 Safety Layer
All outputs validated before dispatch. Interlocks:
- Overcharge protection (SoC > 95%)
- Deep discharge protection (SoC < 10%)  
- Thermal runaway lockout (T > 55°C)
- Anti-islanding (grid voltage/frequency)

---
📦 [GitHub Repository](https://github.com/nikhilrawat2005/energy-management-system) · 
Built with ❤️ using Python, FastAPI, Streamlit, Plotly
""")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # init session state
    for key in ["last_result", "last_state", "sim_df"]:
        if key not in st.session_state:
            st.session_state[key] = None

    coord = get_coordinator()
    page, mode, sim_steps = render_sidebar()

    if page == "🏠 Live Control":
        page_live_control(coord, mode)
    elif page == "📊 Day Simulation":
        page_day_simulation(coord, mode, sim_steps)
    elif page == "🤖 Agent Inspector":
        page_agent_inspector(coord)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
