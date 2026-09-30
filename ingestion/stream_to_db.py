"""
Backfill recent telemetry into TimescaleDB.

This is a thin wrapper around :mod:`ingestion.telemetry` for ad-hoc backfills
(e.g. the last 60 points). It speaks the same metric contract as the live
streamer and the seeder, so no panel ever shows "No data" because of a name
mismatch.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone

import project_paths
from coordinator.coordinator import MultiAgentCoordinator
from coordinator.rl_env import resolve_power_flow
from ingestion.telemetry import TelemetryWriter


SIM_STEP_HOURS = 0.25


def stream_telemetry_to_timescale(
    dataset_path: str = project_paths.DEFAULT_DATASET_PATH,
    host: str = "localhost",
    port: int = 5434,
    points: int = 60,
    verbose: bool = True,
) -> None:
    writer = TelemetryWriter(host=host, port=port, verbose=verbose)
    try:
        writer.connect()
    except RuntimeError as exc:
        if verbose:
            print(f"[DB Feeder] {exc}")
        return

    writer.apply_schema()

    with open(project_paths.resolve(dataset_path), "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    coord = MultiAgentCoordinator()
    limits = project_paths.load_limits()
    batt_cfg = limits.get("battery", {})
    grid_cfg = limits.get("grid", {})
    safety_cfg = limits.get("safety", {})

    cap_kwh = float(batt_cfg.get("capacity_kwh", 15.0))
    pmax_kw = float(grid_cfg.get("pmax_kw", 5.0))
    eff = float(batt_cfg.get("efficiency", 0.95))
    soc_min = float(batt_cfg.get("soc_min", 0.10))
    soc_max = float(batt_cfg.get("soc_max", 0.95))
    soc_init = float(batt_cfg.get("soc_init", 0.50))
    dt = float(SIM_STEP_HOURS)

    soc = soc_init
    total_savings_inr = 0.0
    accumulated_solar_kwh = 0.0
    accumulated_load_kwh = 0.0
    accumulated_grid_kwh = 0.0

    if verbose:
        print(f"[DB Feeder] Backfilling last {points} points from {dataset_path}...")

    for i in range(len(rows) - points, len(rows)):
        r = rows[i]
        now = datetime.now(timezone.utc)

        solar_kw = float(r["solar_kw"])
        load_kw = float(r["load_kw"])
        price = float(r["price"])
        grid_status = int(r["grid_status"])
        batt_temp_c = float(r["batt_temp"])
        solar_fc_1h = float(r.get("solar_fc_1h", solar_kw))
        load_fc_1h = float(r.get("load_fc_1h", load_kw))
        hour = float(r["hour"])

        r_state = {
            "solar_kw": solar_kw,
            "load_kw": load_kw,
            "soc": soc,
            "batt_temp": batt_temp_c,
            "price": price,
            "grid_status": grid_status,
            "grid_voltage": float(r["grid_voltage"]),
            "grid_freq": float(r["grid_freq"]),
            "hour": hour,
            "irradiance_wm2": float(r["irradiance_wm2"]),
            "solar_fc_1h": solar_fc_1h,
            "load_fc_1h": load_fc_1h,
        }
        res = coord.process_tick(r_state, mode="rule_based", current_time_sec=time.time())
        final_action = res["final_action"]
        final_name = res["final_action_name"]
        safety_status = res["safety_status"]
        proposed_action = res.get("proposed_action", final_action)
        safety_override = res.get("safety_override", False)
        decision_source = res.get("decision_source", "rules")

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

        accumulated_solar_kwh += solar_kw * dt
        accumulated_load_kwh += load_kw * dt
        accumulated_grid_kwh += grid_import_kwh

        solar_v = round(380.0, 1) if solar_kw > 0.05 else 0.0
        solar_a = round((solar_kw * 1000.0) / solar_v, 2) if solar_v > 0 else 0.0

        grid_v = round(float(r["grid_voltage"]), 1)
        grid_f = round(float(r["grid_freq"]), 2)
        grid_pf = round(0.96, 3)
        load_a = round((load_kw * 1000.0) / (grid_v * grid_pf), 2)

        batt_v = round(48.0 + (soc * 6.0), 2)
        batt_a = round((batt_kw * 1000.0) / batt_v, 2) if batt_v > 0 else 0.0

        grid_a = round((grid_import_kw * 1000.0) / (grid_v * grid_pf), 2) if grid_status == 1 else 0.0

        grid_only_cost = load_kw * price * dt
        maems_cost = grid_import_kw * price * dt
        total_savings_inr += (grid_only_cost - maems_cost)

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
            "solar_irradiance": float(r["irradiance_wm2"]),
            "soc": soc,
            "soc_pct": round(soc * 100.0, 2),
            "bms_voltage_v": batt_v,
            "bms_current_a": batt_a,
            "bms_power_kw": round(batt_kw, 2),
            "batt_temp_c": batt_temp_c,
            "ambient_temp_c": float(r["ambient_temp"]),
            "humidity_pct": float(r["humidity"]),
            "wind_speed_ms": 2.5,
            "cloud_cover_pct": 15.0,
            "weather_status": 0,
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

        writer.write_snapshot(now, snapshot)
        writer.write_decision(
            ts=now,
            mode="rule_based",
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

    writer.close()
    if verbose:
        print(f"[DB Feeder] Backfill complete. "
              f"sensor_readings: {writer.count('sensor_readings')}")


if __name__ == "__main__":
    stream_telemetry_to_timescale()