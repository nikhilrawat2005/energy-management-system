import psycopg2
conn = psycopg2.connect(host='localhost', port=5434, user='maems_user', password='maems_password', dbname='maems_energy')
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS ai_reasoning_log (
    ts TIMESTAMPTZ NOT NULL,
    action_name TEXT,
    solar_kw DOUBLE PRECISION,
    load_kw DOUBLE PRECISION,
    soc DOUBLE PRECISION,
    price DOUBLE PRECISION,
    tariff_tier TEXT,
    safety_status TEXT,
    reasoning TEXT,
    source TEXT,
    model TEXT,
    latency_ms DOUBLE PRECISION
);
""")
cur.execute("SELECT create_hypertable('ai_reasoning_log', 'ts', if_not_exists => TRUE);")
conn.commit()
print('ai_reasoning_log hypertable created!')
conn.close()
