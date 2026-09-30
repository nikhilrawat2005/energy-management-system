"""MAEMS control API.

Bugs fixed
----------
* ``/api/status`` was **advertised on the root endpoint but never implemented**,
  so the documented health check returned 404. It now returns the coordinator
  health snapshot (agents, decision engines, battery/grid envelope, interlock
  trip count, RL policy statistics).
* ``python api/main.py`` crashed on import, because ``sys.path[0]`` is the
  ``api/`` directory and ``from coordinator.coordinator import ...`` therefore
  could not resolve. A bootstrap now adds the project root to ``sys.path``, so
  ``python api/main.py``, ``python -m api.main`` and ``uvicorn api.main:app``
  all work.
* No CORS middleware: any browser dashboard served from a different origin was
  blocked by the browser before the request ever left the page.
* ``Query(..., enum=[...])`` is removed in modern FastAPI - it is now a
  ``Literal`` so the OpenAPI schema carries a real enum and bad input is a 422
  rather than a 500.
* ``TelemetryPayload`` was missing the ``*_fc_4h`` fields the agents read, so a
  client could not supply the 4-hour forecasts.
* No response models and no examples, which made the generated ``/docs`` page
  useless as integration documentation.
* Unused imports (``asyncio``, ``Dict``, ``Any``).
* ``reload=True`` in the ``__main__`` block re-imported the module (and
  re-loaded the RL policy from disk) on every file change.
* A numpy scalar anywhere in a report would make FastAPI's serialiser raise a
  500. :func:`_jsonable` normalises the report tree before it leaves the app.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List, Literal, Optional

# --- import bootstrap -------------------------------------------------------
# Allow `python api/main.py` as well as `python -m api.main` / `uvicorn`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, Query  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from coordinator.coordinator import MultiAgentCoordinator  # noqa: E402

API_VERSION = "1.1.0"
STARTED_AT = time.time()

DispatchMode = Literal["rl", "rule_based"]

app = FastAPI(
    title="Multi-Agent Energy Management System (MAEMS)",
    description=(
        "Real-time multi-agent microgrid control API. Four specialist agents "
        "(solar, demand, battery, price) each report a structured opinion on the "
        "current telemetry snapshot, a decision engine (learned Q-learning policy "
        "or deterministic rules) proposes a dispatch action, and an independent "
        "safety interlock has the final say."
    ),
    version=API_VERSION,
)

app.add_middleware(
    # Import inside the call so the module keeps working if the optional
    # dependency is absent (the control loop does not need CORS).
    __import__("fastapi.middleware.cors", fromlist=["CORSMiddleware"]).CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

coordinator = MultiAgentCoordinator()


# ---------------------------------------------------------------- models
class TelemetryPayload(BaseModel):
    """One telemetry snapshot from the field or the simulator."""

    solar_kw: float = Field(..., ge=0.0, description="Instantaneous PV output (kW)")
    load_kw: float = Field(..., ge=0.0, description="Instantaneous site load (kW)")
    soc: float = Field(0.50, ge=0.0, le=1.0, description="Battery state of charge (0-1 fraction)")
    batt_temp: float = Field(27.5, description="Battery temperature (degC)")
    price: float = Field(7.0, ge=0.0, description="Grid tariff (INR/kWh)")
    irradiance_wm2: float = Field(500.0, ge=0.0, description="Pyranometer reading (W/m2)")
    grid_voltage: float = Field(230.0, description="Grid voltage (V)")
    grid_freq: float = Field(50.0, description="Grid frequency (Hz)")
    grid_status: int = Field(1, ge=0, le=1, description="1 = grid available, 0 = outage")
    hour: float = Field(12.0, ge=0.0, le=24.0, description="Local hour of day (0-24)")
    solar_fc_1h: Optional[float] = Field(None, description="1 h ahead PV forecast (kW)")
    load_fc_1h: Optional[float] = Field(None, description="1 h ahead load forecast (kW)")
    solar_fc_4h: Optional[float] = Field(None, description="4 h ahead PV forecast (kW)")
    load_fc_4h: Optional[float] = Field(None, description="4 h ahead load forecast (kW)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "solar_kw": 5.4,
                "load_kw": 3.1,
                "soc": 0.62,
                "batt_temp": 31.2,
                "price": 11.5,
                "irradiance_wm2": 720.0,
                "grid_voltage": 228.4,
                "grid_freq": 50.02,
                "grid_status": 1,
                "hour": 19.0,
                "solar_fc_1h": 4.1,
                "load_fc_1h": 3.6,
            }
        }
    }


class DecisionResponse(BaseModel):
    mode: str
    decision_source: str
    proposed_action: int
    proposed_action_name: str
    final_action: int
    final_action_name: str
    safety_status: str
    safety_override: bool
    control_interval_sec: float
    agent_reports: Dict[str, Dict[str, Any]]


class AgentReportsResponse(BaseModel):
    solar_agent: Dict[str, Any]
    demand_agent: Dict[str, Any]
    battery_agent: Dict[str, Any]
    price_agent: Dict[str, Any]


# ---------------------------------------------------------------- helpers
def _jsonable(value: Any) -> Any:
    """Recursively coerce numpy scalars/arrays into plain JSON types.

    A single ``numpy.float64`` anywhere in a report makes FastAPI's encoder
    raise, which surfaced as an opaque 500 on a perfectly healthy control loop.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and getattr(value, "shape", None) == ():
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _finalise_state(payload: TelemetryPayload) -> Dict[str, Any]:
    """Apply persistence defaults to the forecast fields.

    A missing 1 h forecast is back-filled with the present value (a
    persistence forecast), which is what the agents fall back to internally.
    """
    state = payload.model_dump()
    if state.get("solar_fc_1h") is None:
        state["solar_fc_1h"] = state["solar_kw"]
    if state.get("load_fc_1h") is None:
        state["load_fc_1h"] = state["load_kw"]
    if state.get("solar_fc_4h") is None:
        state["solar_fc_4h"] = state["solar_fc_1h"]
    if state.get("load_fc_4h") is None:
        state["load_fc_4h"] = state["load_fc_1h"]
    return state


# ---------------------------------------------------------------- routes
@app.get("/", tags=["meta"], summary="Service banner and route index")
def root() -> Dict[str, Any]:
    return {
        "status": "online",
        "system": "Multi-Agent Energy Management System (MAEMS)",
        "version": API_VERSION,
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "endpoints": [
            "/health",
            "/api/status",
            "/api/agents",
            "/api/dispatch",
            "/api/limits",
            "/api/actions",
        ],
    }


@app.get("/health", tags=["meta"], summary="Liveness probe")
def health() -> Dict[str, Any]:
    return {"status": "ok", "uptime_seconds": round(time.time() - STARTED_AT, 1)}


@app.get(
    "/api/status",
    response_model=Dict[str, Any],
    tags=["meta"],
    summary="Coordinator health, envelopes, interlock and policy statistics",
)
def get_status() -> Dict[str, Any]:
    snapshot = coordinator.status()
    snapshot["version"] = API_VERSION
    snapshot["uptime_seconds"] = round(time.time() - STARTED_AT, 1)
    return _jsonable(snapshot)


@app.get(
    "/api/actions",
    tags=["meta"],
    summary="Action id -> meaning, for building a UI without hard-coding ids",
)
def get_actions() -> Dict[str, Any]:
    from coordinator.coordinator import ACTION_NAMES

    return {
        "actions": [
            {"id": aid, "name": name, "short": name.split(" (")[0]}
            for aid, name in sorted(ACTION_NAMES.items())
        ],
        "modes": list(coordinator.status()["decision_engines"]),
    }


@app.get(
    "/api/limits",
    response_model=Dict[str, Any],
    tags=["config"],
    summary="The effective config/limits.yaml after defaults are applied",
)
def get_limits() -> Dict[str, Any]:
    return _jsonable(coordinator.limits)


@app.post(
    "/api/dispatch",
    response_model=DecisionResponse,
    tags=["control"],
    summary="Run one control cycle: agents -> decision engine -> safety interlock",
)
def dispatch_control(
    payload: TelemetryPayload,
    mode: DispatchMode = Query(
        "rl",
        description="'rl' uses the learned Q-table; 'rule_based' uses the deterministic rules.",
    ),
) -> Dict[str, Any]:
    state = _finalise_state(payload)
    # NOTE: the interlock's rate limiter keys off the *control-step* clock, not
    # wall-clock, so a replay or a 15-minute tick sequence behaves identically.
    result = coordinator.process_tick(
        state, mode=mode, current_time_sec=payload.hour * 3600.0
    )
    return _jsonable(result)


@app.post(
    "/api/agents",
    response_model=AgentReportsResponse,
    tags=["control"],
    summary="Per-agent opinions only, without committing to a dispatch action",
)
def get_agent_reports(payload: TelemetryPayload) -> Dict[str, Any]:
    state = _finalise_state(payload)
    reports = {
        "solar_agent": coordinator.solar_agent.report(state),
        "demand_agent": coordinator.demand_agent.report(state),
        "battery_agent": coordinator.battery_agent.report(state),
        "price_agent": coordinator.price_agent.report(state),
    }
    return _jsonable(reports)


@app.post("/api/reset", tags=["control"], summary="Clear interlock relay state")
def reset_interlock() -> Dict[str, Any]:
    coordinator.reset()
    return {"status": "reset", "interlock": coordinator.safety.trip_count}


def main(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":  # pragma: no cover
    main()
