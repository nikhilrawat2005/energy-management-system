import asyncio
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time

from coordinator.coordinator import MultiAgentCoordinator

app = FastAPI(
    title="Multi-Agent Energy Management System (MAEMS)",
    description="Real-time multi-agent microgrid control API with RL, Rule-based coordinator & safety layer.",
    version="1.0.0"
)

coordinator = MultiAgentCoordinator()

class TelemetryPayload(BaseModel):
    solar_kw: float
    load_kw: float
    soc: float = 0.50
    batt_temp: float = 27.5
    price: float = 7.0
    irradiance_wm2: float = 500.0
    grid_voltage: float = 230.0
    grid_freq: float = 50.0
    grid_status: int = 1
    hour: float = 12.0
    solar_fc_1h: Optional[float] = None
    load_fc_1h: Optional[float] = None

@app.get("/")
def root():
    return {
        "status": "online",
        "system": "Multi-Agent Energy Management System (MAEMS)",
        "docs_url": "/docs",
        "endpoints": ["/api/status", "/api/agents", "/api/dispatch", "/api/limits"]
    }

@app.get("/api/limits")
def get_limits():
    return coordinator.limits

@app.post("/api/dispatch")
def dispatch_control(payload: TelemetryPayload, mode: str = Query("rl", enum=["rl", "rule_based"])):
    state = payload.model_dump()
    if state["solar_fc_1h"] is None:
        state["solar_fc_1h"] = state["solar_kw"]
    if state["load_fc_1h"] is None:
        state["load_fc_1h"] = state["load_kw"]
        
    result = coordinator.process_tick(state, mode=mode, current_time_sec=time.time())
    return result

@app.post("/api/agents")
def get_agent_reports(payload: TelemetryPayload):
    state = payload.model_dump()
    solar_rep = coordinator.solar_agent.report(state)
    demand_rep = coordinator.demand_agent.report(state)
    battery_rep = coordinator.battery_agent.report(state)
    price_rep = coordinator.price_agent.report(state)
    return {
        "solar_agent": solar_rep,
        "demand_agent": demand_rep,
        "battery_agent": battery_rep,
        "price_agent": price_rep
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
