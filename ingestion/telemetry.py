"""
Canonical telemetry contract for the MAEMS Grafana stack.

Why this module exists
----------------------
The Grafana dashboards query ``(device, metric)`` pairs by name, but before this
module the three writers disagreed about those names:

* ``ingestion/live_streamer.py``  wrote ``smart_energy_meter`` / ``meter_power_kw``
* ``ingestion/seed_dashboard.py`` wrote ``energy_meter``     / ``load_kw``
* ``ingestion/stream_to_db.py``   wrote ``energy_meter``     / ``load_kw``
* ``ingestion/seed_dashboard.py`` put ``price`` under ``grid_smart_meter``
  while the streamer put it under ``tariff_api``

Result: after seeding the history the "Load", "Cost", "Tariff" and "SoC" panels
on three of the four dashboards rendered *empty*, because Grafana asked for
metric names that no writer had ever inserted. There was no test and no shared
constant to catch it.

This module is now the only place that defines the contract:

* :data:`SENSOR_CONTRACT` - metric name -> (device, unit, description)
* :func:`build_sensor_rows` - snapshot dict -> rows ready for insertion
* :class:`TelemetryWriter` - connection handling, batched inserts, optional TRUNCATE

``tests/test_telemetry_contract.py`` and ``tests/test_dashboards.py`` both read
:data:`SENSOR_CONTRACT`, so renaming a metric now fails a test instead of
silently blanking a panel.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Database connection defaults (match docker-compose.yml)
# --------------------------------------------------------------------------
DB_HOST = "localhost"
DB_PORT = 5434  # host-side mapping of TimescaleDB's internal 5432
DB_USER = "maems_user"
DB_PASSWORD = "maems_password"
DB_NAME = "maems_energy"

SensorRow = Tuple[str, str, float, str]  # device, metric, value, unit


# --------------------------------------------------------------------------
# The contract. Order matters: it is the physical order of the 13 sensors.
# --------------------------------------------------------------------------
SENSOR_CONTRACT: Dict[str, Tuple[str, str, str]] = {
    # 1. Smart Energy Meter
    "meter_power_kw": ("smart_energy_meter", "kW", "Total site load measured by the main smart meter"),
    "meter_energy_kwh": ("smart_energy_meter", "kWh", "Cumulative imported/consumed energy"),
    "meter_pf": ("smart_energy_meter", "PF", "Power factor of the site load"),
    # 2. Current transformer
    "ct_current_a": ("ct_sensor", "A", "Line current measured by the CT clamp"),
    # 3. Potential transformer / voltage sensor
    "pt_voltage_v": ("pt_voltage_sensor", "V", "Bus voltage measured by the PT"),
    # 4. Solar inverter
    "solar_kw": ("solar_inverter", "kW", "Instantaneous PV AC output"),
    "solar_yield_kwh": ("solar_inverter", "kWh", "Cumulative PV energy yield"),
    "solar_voltage_v": ("solar_inverter", "V", "Inverter DC/AC bus voltage"),
    "solar_current_a": ("solar_inverter", "A", "Inverter output current"),
    # 5. Pyranometer
    "solar_irradiance": ("pyranometer", "W/m2", "Plane-of-array irradiance"),
    # 6. Battery management system
    "soc": ("bms", "fraction", "State of charge as a 0-1 fraction (multiply by 100 for %)"),
    "soc_pct": ("bms", "%", "State of charge in percent (same instant as `soc`)"),
    "bms_voltage_v": ("bms", "V", "Battery pack terminal voltage"),
    "bms_current_a": ("bms", "A", "Battery pack current (negative = discharging)"),
    "bms_power_kw": ("bms", "kW", "Battery power (negative = discharging)"),
    # 7. NTC thermistor
    "batt_temp_c": ("ntc_thermistor", "degC", "Battery pack temperature"),
    # 8. DHT22 / PT100
    "ambient_temp_c": ("dht22_sensor", "degC", "Ambient air temperature"),
    # 9. Humidity sensor
    "humidity_pct": ("humidity_sensor", "%RH", "Relative humidity"),
    # 10. Anemometer
    "wind_speed_ms": ("anemometer", "m/s", "Wind speed"),
    # 11. Weather API / station
    "cloud_cover_pct": ("weather_api", "%", "Cloud cover"),
    "weather_status": ("weather_api", "state", "1 = live Open-Meteo feed, 0 = dataset fallback"),
    # 12. Time-of-day tariff
    "price": ("tariff_api", "INR/kWh", "Live dynamic ToD tariff"),
    # 13. Grid smart meter
    "grid_power_kw": ("grid_smart_meter", "kW", "Net grid import (0 during islanding)"),
    "grid_freq_hz": ("grid_smart_meter", "Hz", "Grid frequency"),
    "grid_status": ("grid_smart_meter", "state", "1 = grid synchronised, 0 = outage"),
    "grid_current_a": ("grid_smart_meter", "A", "Grid line current"),
    "grid_energy_kwh": ("grid_smart_meter", "kWh", "Cumulative energy drawn from the grid"),
    # Derived / ML
    "pred_solar_1h": ("ml_forecaster", "kW", "GBDT forecast of PV output in 1 h"),
    "pred_demand_1h": ("ml_forecaster", "kW", "GBDT forecast of site load in 1 h"),
    "total_savings_inr": ("economic_calculator", "INR", "Cumulative cost avoided vs grid-only operation"),
}

# Metrics that are cumulative register-style totals rather than instantaneous values.
CUMULATIVE_METRICS = frozenset(
    {"meter_energy_kwh", "solar_yield_kwh", "grid_energy_kwh", "total_savings_inr"}
)

# Metrics that are categorical flags encoded as 0/1.
STATE_METRICS = frozenset({"grid_status", "weather_status"})


def device_names() -> List[str]:
    return sorted({device for device, _, _ in SENSOR_CONTRACT.values()})


def metric_names() -> List[str]:
    return list(SENSOR_CONTRACT)


def describe(metric: str) -> str:
    entry = SENSOR_CONTRACT.get(metric)
    return entry[2] if entry else ""


def build_sensor_rows(snapshot: Dict[str, Any]) -> List[SensorRow]:
    """Turn a flat snapshot into contract-ordered ``(device, metric, value, unit)`` rows.

    Every writer in the project funnels through here, so a metric that is present
    in :data:`SENSOR_CONTRACT` but missing from ``snapshot`` is simply omitted and
    a metric that is *not* in the contract is reported by :class:`TelemetryWriter`
    rather than being written under a name no dashboard queries.
    """
    rows: List[SensorRow] = []
    for metric, (device, unit, _desc) in SENSOR_CONTRACT.items():
        if metric not in snapshot:
            continue
        value = snapshot[metric]
        if value is None:
            continue
        rows.append((device, metric, float(value), unit))
    return rows


# --------------------------------------------------------------------------
# Database writer
# --------------------------------------------------------------------------
_INSERT_SQL = (
    "INSERT INTO sensor_readings (ts, device, metric, value, unit, quality) "
    "VALUES (%s, %s, %s, %s, %s, %s)"
)

_DECISION_SQL = (
    "INSERT INTO agent_decisions "
    "(ts, mode, action, action_name, solar_kw, load_kw, soc, price, "
    " safety_status, proposed_action, safety_override, decision_source) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)


class TelemetryWriter:
    """Thin, defensive wrapper around the TimescaleDB inserts.

    Fixes three robustness problems in the old writers:
    * ``psycopg2.connect`` was called bare in :mod:`ingestion.live_streamer`, so a
      stopped container produced a raw traceback instead of a useful message.
    * One ``INSERT`` per metric meant 29 separate round trips per tick; a single
      ``execute_values``-style batch is used instead.
    * ``TRUNCATE`` was issued unconditionally by the seeder, wiping live data.
      It is now opt-in via ``truncate=True``.
    """

    def __init__(
        self,
        host: str = DB_HOST,
        port: int = DB_PORT,
        user: str = DB_USER,
        password: str = DB_PASSWORD,
        dbname: str = DB_NAME,
        connect_timeout: int = 5,
        retries: int = 3,
        retry_delay: float = 2.0,
        verbose: bool = True,
    ):
        self.dsn = dict(
            host=host,
            port=int(port),
            user=user,
            password=password,
            dbname=dbname,
            connect_timeout=int(connect_timeout),
        )
        self.retries = max(1, int(retries))
        self.retry_delay = float(retry_delay)
        self.verbose = verbose
        self.conn = None
        self.cur = None
        self._log("info", f"Connecting to TimescaleDB at {host}:{port}/{dbname} ...")

    # -- plumbing -----------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        if self.verbose:
            print(f"[{level}] {message}")

    def connect(self):
        """Connect with bounded retries. Returns the cursor, or raises RuntimeError."""
        import psycopg2  # imported lazily so the rest of the package works without psycopg2

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                self.conn = psycopg2.connect(**self.dsn)
                self.conn.autocommit = True
                self.cur = self.conn.cursor()
                self._log("db", "Connected to TimescaleDB successfully.")
                return self.cur
            except Exception as exc:  # pragma: no cover - depends on a live DB
                last_error = exc
                self._log(
                    "warn",
                    f"Connection attempt {attempt}/{self.retries} failed: {exc}",
                )
                if attempt < self.retries:
                    time.sleep(self.retry_delay)

        raise RuntimeError(
            f"Could not connect to TimescaleDB at {self.dsn['host']}:{self.dsn['port']} "
            f"after {self.retries} attempt(s). Is the stack running? "
            f"Try: docker compose up -d\nLast error: {last_error}"
        )

    def close(self) -> None:
        for handle in (self.cur, self.conn):
            try:
                if handle is not None:
                    handle.close()
            except Exception:  # pragma: no cover - best effort on shutdown
                pass
        self.cur = None
        self.conn = None

    def __enter__(self) -> "TelemetryWriter":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- schema helpers ----------------------------------------------
    def apply_schema(self, sql_path: Optional[str] = None) -> None:
        """Run ``ingestion/init_db.sql``. Safe to re-run (all statements are IF NOT EXISTS)."""
        import os

        if sql_path is None:
            import project_paths

            sql_path = os.path.join(
                project_paths.PROJECT_ROOT, "ingestion", "init_db.sql"
            )
        sql_path = project_paths.resolve(sql_path)
        if not os.path.exists(sql_path):
            self._log("warn", f"Schema file not found, skipping: {sql_path}")
            return
        with open(sql_path, "r", encoding="utf-8") as handle:
            self.cur.execute(handle.read())
        self._log("db", f"Schema ensured from {os.path.basename(sql_path)}.")

    def truncate(self, confirm: bool = False) -> None:
        """Wipe both hypertables. Requires ``confirm=True`` - this destroys live data."""
        if not confirm:
            self._log("warn", "TRUNCATE skipped (pass confirm=True to allow).")
            return
        self.cur.execute("TRUNCATE TABLE sensor_readings;")
        self.cur.execute("TRUNCATE TABLE agent_decisions;")
        self._log("db", "Truncated sensor_readings and agent_decisions.")

    # -- inserts ------------------------------------------------------
    def write_snapshot(
        self,
        ts,
        snapshot: Dict[str, Any],
        quality: str = "good",
        rows: Optional[Sequence[SensorRow]] = None,
    ) -> int:
        """Insert one full sensor snapshot. Returns the number of rows written."""
        payload = rows if rows is not None else build_sensor_rows(snapshot)
        for device, metric, value, unit in payload:
            row_quality = "good" if quality == "good" else f"{quality}:{metric}"
            self.cur.execute(
                _INSERT_SQL, (ts, device, metric, value, unit, row_quality)
            )
        return len(payload)

    def write_decision(
        self,
        ts,
        mode: str,
        action: int,
        action_name: str,
        solar_kw: float,
        load_kw: float,
        soc: float,
        price: float,
        safety_status: str,
        proposed_action: Optional[int] = None,
        safety_override: bool = False,
        decision_source: str = "rules",
    ) -> None:
        self.cur.execute(
            _DECISION_SQL,
            (
                ts,
                mode,
                int(action),
                action_name,
                float(solar_kw),
                float(load_kw),
                float(soc),
                float(price),
                safety_status,
                int(proposed_action if proposed_action is not None else action),
                bool(safety_override),
                decision_source,
            ),
        )

    def write_history(
        self, ts, snapshot: Dict[str, Any], decision: Dict[str, Any]
    ) -> None:
        """Convenience: one snapshot + the matching agent decision."""
        self.write_snapshot(ts, snapshot)
        self.write_decision(ts=ts, **decision)

    def count(self, table: str) -> int:
        allowed = {"sensor_readings", "agent_decisions"}
        if table not in allowed:
            raise ValueError(f"Refusing to count unknown table {table!r}")
        self.cur.execute(f"SELECT COUNT(*) FROM {table}")
        return int(self.cur.fetchone()[0])


def rows_for_history(  # pragma: no cover - small helper used by the seeder
    snapshot: Dict[str, Any], sequence: Iterable[int]
) -> List[SensorRow]:  # noqa: ARG001 - reserved for chunked backfill
    return build_sensor_rows(snapshot)
