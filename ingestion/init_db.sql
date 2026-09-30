CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE TABLE IF NOT EXISTS sensor_readings (
    ts TIMESTAMPTZ NOT NULL,
    device TEXT NOT NULL,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit TEXT,
    quality TEXT DEFAULT 'good'
);

SELECT create_hypertable('sensor_readings', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_device_metric ON sensor_readings (device, metric, ts DESC);

CREATE TABLE IF NOT EXISTS agent_decisions (
    ts TIMESTAMPTZ NOT NULL,
    mode TEXT NOT NULL,
    action INT NOT NULL,
    action_name TEXT NOT NULL,
    solar_kw DOUBLE PRECISION,
    load_kw DOUBLE PRECISION,
    soc DOUBLE PRECISION,
    price DOUBLE PRECISION,
    safety_status TEXT
);

SELECT create_hypertable('agent_decisions', 'ts', if_not_exists => TRUE);
