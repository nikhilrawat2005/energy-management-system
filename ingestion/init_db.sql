-- MAEMS TimescaleDB schema
-- Safe to re-run (all statements are IF NOT EXISTS)

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Core sensor readings hypertable
CREATE TABLE IF NOT EXISTS sensor_readings (
    ts TIMESTAMPTZ NOT NULL,
    device TEXT NOT NULL,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit TEXT,
    quality TEXT DEFAULT 'good'
);

SELECT create_hypertable('sensor_readings', 'ts', if_not_exists => TRUE);

-- Fast lookups for Grafana panels: WHERE device='x' AND metric='y' ORDER BY ts DESC LIMIT N
CREATE INDEX IF NOT EXISTS idx_device_metric_ts
    ON sensor_readings (device, metric, ts DESC);

-- Agent decisions hypertable
CREATE TABLE IF NOT EXISTS agent_decisions (
    ts TIMESTAMPTZ NOT NULL,
    mode TEXT NOT NULL,
    action INT NOT NULL,
    action_name TEXT NOT NULL,
    solar_kw DOUBLE PRECISION,
    load_kw DOUBLE PRECISION,
    soc DOUBLE PRECISION,
    price DOUBLE PRECISION,
    safety_status TEXT,
    proposed_action INT,
    safety_override BOOLEAN DEFAULT FALSE,
    decision_source TEXT DEFAULT 'rules'
);

SELECT create_hypertable('agent_decisions', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_decision_mode_ts
    ON agent_decisions (mode, ts DESC);

-- Compression policies (TimescaleDB 2.0+) - keep 7d raw, then compress older chunks
ALTER TABLE sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device, metric'
);
ALTER TABLE agent_decisions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'mode, action'
);

-- Retention: 90 days raw data, then drop
SELECT add_retention_policy('sensor_readings', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('agent_decisions', INTERVAL '90 days', if_not_exists => TRUE);

-- Continuous aggregate: 1-min downsample of key metrics for fast long-range queries
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', ts) AS bucket,
    device,
    metric,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    COUNT(*) AS n_samples,
    FIRST(value, ts) AS first_value,
    LAST(value, ts) AS last_value
FROM sensor_readings
GROUP BY bucket, device, metric
WITH NO DATA;

-- Refresh policy: materialize the last 2 hours every 1 minute
SELECT add_continuous_aggregate_policy('sensor_readings_1m',
    start_offset => INTERVAL '2 hours',
    end_offset   => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- Useful views for the dashboards (stable metric/device names per contract)
CREATE OR REPLACE VIEW v_latest_sensor AS
SELECT DISTINCT ON (device, metric)
    device, metric, ts, value, unit, quality
FROM sensor_readings
ORDER BY device, metric, ts DESC;

CREATE OR REPLACE VIEW v_latest_decision AS
SELECT *
FROM agent_decisions
ORDER BY ts DESC
LIMIT 1;