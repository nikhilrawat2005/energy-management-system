"""
Unified Live Streamer - the single writer for the 13-sensor Grafana stack.

All telemetry published to TimescaleDB flows through :mod:`ingestion.telemetry`,
so the dashboard panels and the CSV dataset always speak the same metric names.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone

import numpy as np

import project_paths
from coordinator.coordinator import MultiAgentCoordinator
from forecasting.features import SimpleGradientBoostingRegressor
from ingestion.api_fetcher import WeatherAPIFetcher
from ingestion.telemetry import TelemetryWriter, build_sensor_rows
from coordinator.rl_env import resolve_power_flow


SIM_STEP_HOURS = 0.25  # 15 min control interval used by the env, evaluator, and streamer


def run_unified_streamer(
    dataset_path: str = project_paths.DEFAULT_DATASET_PATH,
    wall_interval_sec: float = 2.0,
    max_steps: Optional[int] = None,
    mode: str = "rule_based",
    verbose: bool = True,
) -> None:
    """Stream the 13-sensor telemetry + forecasts + agent decisions to TimescaleDB.

    Parameters
    ----------
    dataset_path: CSV with the 30-day 15-min dataset (2880 rows).
    wall_interval_sec: wall-clock sleep between emitted steps (default 2 s for a lively demo).
    max_steps: optional cap on iterations; ``None`` runs forever. Useful for tests / CI.
    mode: ``"rule_based"`` or ``"rl"`` - which coordinator mode to run.
    """
    # --- DB --------------------------------------------------------------
    writer = TelemetryWriter(verbose=verbose)
    try:
        writer.connect()
    except RuntimeError as exc:
        if verbose:
            print(f"[Streamer] {exc}")
        return

    # Ensure schema (idempotent)
    writer.apply_schema()

    # --- Models & coordinator -------------------------------------------
    coord = MultiAgentCoordinator()
    weather_fetcher = WeatherAPIFetcher()

    demand_model = SimpleGradientBoostingRegressor()
    solar_model = SimpleGradientBoostingRegressor()
    demand_model.load(str(project_paths.DEMAND_MODEL_PATH))
    solar_model.load(str(project_paths.SOLAR_MODEL_PATH))
    if verbose:
        print("[Streamer] Models loaded, weather fetcher ready.")

    # --- Dataset ---------------------------------------------------------
    with open(project_paths.resolve(dataset_path), "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # --- Config from limits.yaml ----------------------------------------
    limits = project_paths.load_limits()
    batt_cfg = limits.get("battery", {})
    grid_cfg = limits.get("grid", {})
    safety_cfg = limits.get("safety", {})
    tariff_cfg = limits.get("tariff", {})

    cap_kwh = float(batt_cfg.get("capacity_kwh", 15.0))
    pmax_kw = float(grid_cfg.get("pmax_kw", 5.0))
    eff = float(batt_cfg.get("efficiency", 0.95))
    soc_min = float(batt_cfg.get("soc_min", 0.10))
    soc_max = float(batt_cfg.get("soc_max", 0.95))
    soc_init = float(batt_cfg.get("soc_init", 0.50))
    dt = float(SIM_STEP_HOURS)

    # --- State -----------------------------------------------------------
    soc = soc_init
    step_idx = 0
    total_savings_inr = 0.0
    accumulated_solar_kwh = 0.0
    accumulated_load_kwh = 0.0
    accumulated_grid_kwh = 0.0
    last_weather_fetch = 0.0
    cached_weather: Dict[str, Any] = {}

    if verbose:
        print(f"[Streamer] Streaming {len(rows)} rows @ {wall_interval_sec}s wall interval, "
              f"SIM_STEP_HOURS={SIM_STEP_HOURS} (control clock = step_idx * 900s)")

    try:
        while max_steps is None or step_idx < max_steps:
            r = rows[step_idx % len(rows)]
            ts = datetime.now(timezone.utc)

            # --- Periodic live weather poll (30 s wall clock) -------------
            if time.time() - last_weather_fetch > 30:
                live_w = weather_fetcher.fetch_current_and_forecast()
                if live_w.get("status") == "online":
                    cached_weather = live_w
                last_weather_fetch = time.time()

            # --- Parse CSV row --------------------------------------------
            hour = float(r["hour"])
            day_of_week = float(r["day_of_week"])
            is_weekend = float(r["is_weekend"])
            solar_kw = float(r["solar_kw"])
            load_kw = float(r["load_kw"])
            price = float(r["price"])
            grid_status = int(r["grid_status"])
            batt_temp_c = float(r["batt_temp"])
            solar_fc_1h = float(r.get("solar_fc_1h", solar_kw))
            load_fc_1h = float(r.get("load_fc_1h", load_kw))

            # --- Prefer live weather; fall back to dataset -----------------
            irradiance_wm2 = float(cached_weather.get("irradiance_wm2", r["irradiance_wm2"]))
            cloud_cover_pct = float(cached_weather.get("cloud_cover_pct", 15.0))
            ambient_temp_c = float(cached_weather.get("ambient_temp", r["ambient_temp"]))
            humidity_pct = float(cached_weather.get("humidity", r["humidity"]))
            weather_status = 1 if cached_weather.get("status") == "online" else 0

            # --- Wind (synthetic, dataset has no wind column) -------------
            wind_speed_ms = round(float(2.5 + 1.8 * np.sin(hour / 3.0) + np.random.normal(0, 0.4)), 2)
            wind_speed_ms = max(0.2, wind_speed_ms)

            # --- Solar inverter derived values -----------------------------
            solar_v = round(380.0 + np.random.normal(0, 3.0), 1) if solar_kw > 0.05 else 0.0
            # Safe: solar_v=0.0 -> solar_a=0.0 (not solar_kw*1000)
            solar_a = round((solar_kw * 1000.0) / solar_v, 2) if solar_v > 0 else 0.0

            # --- Load meter derived values --------------------------------
            grid_v = round(float(r["grid_voltage"]), 1)
            grid_f = round(float(r["grid_freq"]), 2)
            grid_pf = round(float(0.96 + np.random.normal(0, 0.01)), 3)
            load_a = round((load_kw * 1000.0) / (grid_v * grid_pf), 2)

            # --- Coordinator tick (control clock = step_idx * 900) -------
            r_state = {
                "solar_kw": solar_kw,
                "load_kw": load_kw,
                "soc": soc,
                "batt_temp": batt_temp_c,
                "price": price,
                "grid_status": grid_status,
                "grid_voltage": grid_v,
                "grid_freq": grid_f,
                "hour": hour,
                "irradiance_wm2": irradiance_wm2,
                "solar_fc_1h": solar_fc_1h,
                "load_fc_1h": load_fc_1h,
            }
            res = coord.process_tick(r_state, mode=mode, current_time_sec=step_idx * 900.0)
            final_action = res["final_action"]
            final_name = res["final_action_name"]
            safety_status = res["safety_status"]
            proposed_action = res.get("proposed_action", final_action)
            safety_override = res.get("safety_override", False)
            decision_source = res.get("decision_source", "rules")

            # --- Shared power-balance physics (single source of truth) ----
            flow = resolve_power_flow(
                action=final_action,
                solar_kw=solar_kw,
                load_kw=load_kw,
                soc=soc,
                grid_status=grid_status,
                pmax_kw=pmax_kw,
                cap_kwh=cap_kwh,
                eff=eff,
                dt=dt,
                soc_min=soc_min,
                soc_max=soc_max,
            )
            soc = flow["soc"]
            batt_kw = flow["batt_kw"]
            grid_import_kw = flow["grid_import_kw"]
            grid_import_kwh = flow["grid_import_kwh"]
            solar_used_kwh = flow["solar_used_kwh"]
            curtailed_solar_kwh = flow["curtailed_solar_kwh"]

            # Accumulators (kWh)
            accumulated_solar_kwh += solar_kw * dt
            accumulated_load_kwh += load_kw * dt
            accumulated_grid_kwh += grid_import_kwh

            # --- Battery derived readings ---------------------------------
            batt_v = round(48.0 + (soc * 6.0), 2)  # 48 V .. 54 V pack
            batt_a = round((batt_kw * 1000.0) / batt_v, 2) if batt_v > 0 else 0.0

            # --- Grid meter derived readings -------------------------------
            grid_a = round((grid_import_kw * 1000.0) / (grid_v * grid_pf), 2) if grid_status == 1 else 0.0

            # --- Economics -------------------------------------------------
            grid_only_cost = load_kw * price * dt
            maems_cost = grid_import_kw * price * dt
            total_savings_inr += (grid_only_cost - maems_cost)  # no clamp - let it net

            # --- Snapshot for the contract writer -------------------------
            snapshot = {
                "meter_power_kw": load_kw,
                "meter_energy_kwh": round(accumulated_load_kwh, 2),
                "meter_pf": grid_pf,
                "ct_current_a": load_a,
                "pt_voltage_v": grid_v,
                "solar_kw": solar_kw,
                "solar_yield_kwh": round(accumulated_solar_kwh, 2),
                "solar_voltage_v": solar_v,
                "solar_current_a": solar_a,
                "solar_irradiance": irradiance_wm2,
                "soc": soc,
                "soc_pct": round(soc * 100.0, 2),
                "bms_voltage_v": batt_v,
                "bms_current_a": batt_a,
                "bms_power_kw": round(batt_kw, 2),
                "batt_temp_c": batt_temp_c,
                "ambient_temp_c": ambient_temp_c,
                "humidity_pct": humidity_pct,
                "wind_speed_ms": wind_speed_ms,
                "cloud_cover_pct": cloud_cover_pct,
                "weather_status": weather_status,
                "price": price,
                "grid_power_kw": round(grid_import_kw, 2),
                "grid_freq_hz": grid_f,
                "grid_status": grid_status,
                "grid_current_a": grid_a,
                "grid_energy_kwh": round(accumulated_grid_kwh, 2),
                "pred_solar_1h": solar_fc_1h,
                "pred_demand_1h": load_fc_1h,
                "total_savings_inr": round(total_savings_inr, 2),
            }

            # --- Write everything -----------------------------------------
            writer.write_snapshot(ts, snapshot)
            writer.write_decision(
                ts=ts,
                mode=mode,
                action=final_action,
                action_name=final_name,
                solar_kw=solar_kw,
                load_kw=load_kw,
                soc=soc,
                price=price,
                safety_status=safety_status,
                proposed_action=proposed_action,
                safety_override=safety_override,
                decision_source=decision_source,
            )

            # --- AI Agent Reasoning (every 5 steps to keep loop fast) ------
            if step_idx % 5 == 0:
                try:
                    from agents.ai_coordinator_agent import generate_ai_reasoning
                    agent_reports = res.get("agent_reports", {})
                    solar_rep = agent_reports.get("solar", {})
                    demand_rep = agent_reports.get("demand", {})
                    battery_rep = agent_reports.get("battery", {})
                    price_rep = agent_reports.get("price", {})
                    ai_result = generate_ai_reasoning(
                        state=r_state,
                        solar_rep=solar_rep,
                        demand_rep=demand_rep,
                        battery_rep=battery_rep,
                        price_rep=price_rep,
                        action_name=final_name,
                        safety_status=safety_status,
                    )
                    # Write AI reasoning to DB
                    if writer.cur is not None:
                        writer.cur.execute(
                            """INSERT INTO ai_reasoning_log
                               (ts, action_name, solar_kw, load_kw, soc, price, tariff_tier,
                                safety_status, reasoning, source, model, latency_ms)
                               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (
                                ts,
                                final_name,
                                round(solar_kw, 3),
                                round(load_kw, 3),
                                round(soc, 4),
                                round(price, 2),
                                price_rep.get("tariff_tier", "NORMAL"),
                                safety_status,
                                ai_result["reasoning"],
                                ai_result["source"],
                                ai_result["model"],
                                ai_result["latency_ms"],
                            )
                        )
                        writer.conn.commit()
                    if verbose and step_idx % 20 == 0:
                        src = ai_result["source"].upper()
                        lat = ai_result["latency_ms"]
                        print(f"  [AI-{src} {lat:.0f}ms] {ai_result['reasoning'][:100]}...")
                except Exception as ai_err:
                    if verbose:
                        print(f"  [AI] Error: {ai_err}")

            if verbose and step_idx % 20 == 0:
                print(f"  step={step_idx:4d}  action={final_name:12s}  "
                      f"soc={soc:.3f}  grid={grid_import_kw:.2f}kW  "
                      f"savings=Rs.{total_savings_inr:.2f}  safety={safety_status}")

            step_idx += 1
            time.sleep(wall_interval_sec)

    except KeyboardInterrupt:
        if verbose:
            print("\n[Streamer] Interrupted by user - shutting down.")
    except Exception as exc:  # pragma: no cover
        if verbose:
            print(f"[Streamer] Unexpected error: {exc}")
    finally:
        writer.close()


if __name__ == "__main__":
    run_unified_streamer()