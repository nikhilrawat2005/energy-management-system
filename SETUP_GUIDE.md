# Setup Guide: External Services, Docker & API Integration

This guide walks you through setting up Docker services (TimescaleDB, Mosquitto MQTT, Grafana) and Weather/Tariff APIs for the **Multi-Agent Energy Management System (MAEMS)**.

---

## 1. Docker Setup (TimescaleDB, MQTT Broker, Grafana)

The project includes a ready-to-run [`docker-compose.yml`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/docker-compose.yml).

### Step 1: Start Docker Desktop
1. Open **Docker Desktop** on Windows from your Start menu.
2. Wait until the whale icon in the bottom left turns **green** (Engine Running).

### Step 2: Spin Up the Stack
Open PowerShell in the project directory and run:
```powershell
docker compose up -d
```

### Step 3: Verified Ports & Services
| Service | Container Name | Port | Credentials | Purpose |
|---|---|---|---|---|
| **Mosquitto** | `maems_mosquitto` | `1883` | None (Anonymous allowed) | IoT / Telemetry MQTT Message Broker |
| **TimescaleDB** | `maems_timescaledb` | `5432` | User: `maems_user`<br>Pass: `maems_password`<br>DB: `maems_energy` | Real-time Hypertable sensor database |
| **Grafana** | `maems_grafana` | `3000` | User: `admin`<br>Pass: `admin` | Real-time Power & Battery Dashboards |

To stop the containers at any time:
```powershell
docker compose down
```

---

## 2. Weather & Solar Forecast API

We integrated **Open-Meteo Solar API** into [`ingestion/api_fetcher.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/ingestion/api_fetcher.py).

### Why Open-Meteo?
- **Zero API Key Needed**: Works right out of the box without credit cards or account registration.
- **Accurate Solar Radiation**: Provides `direct_normal_irradiance` ($W/m^2$), `cloud_cover` (%), `temperature_2m`, and `relative_humidity_2m`.
- **Live Tested**: Verified live response:
  ```json
  {"ambient_temp": 29.8, "humidity": 62, "irradiance_wm2": 643.0, "cloud_cover_pct": 0}
  ```

### Customizing Coordinates:
If you want weather/solar irradiance for your specific city, edit the coordinates in `ingestion/api_fetcher.py`:
```python
fetcher = WeatherAPIFetcher(lat=28.6139, lon=77.2090) # Change to your latitude & longitude
```

---

## 3. Electricity Tariff (ToD) Setup

Electricity pricing is configured in [`config/limits.yaml`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/config/limits.yaml).

If you want to customize your DISCOM's tariff brackets:
```yaml
tariff:
  currency: "INR"
  off_peak:
    hours: [0, 1, 2, 3, 4, 5, 22, 23]
    rate_per_kwh: 4.5
  normal:
    hours: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    rate_per_kwh: 7.0
  peak:
    hours: [18, 19, 20, 21]
    rate_per_kwh: 11.5
```

---

## 4. Ingesting Data4Cyber Dataset

When you download your Data4Cyber scenario CSV files:
1. Place the CSV file in `data/` (e.g. `data/data4cyber_run1.csv`).
2. Stream and test it directly using [`ingestion/data4cyber_reader.py`](file:///c:/Users/NIKHIL%20RAWAT/OneDrive/Desktop/Nikhil%20PROJECT/energy%20system/ingestion/data4cyber_reader.py):
   ```python
   from ingestion.data4cyber_reader import Data4CyberStreamReader
   reader = Data4CyberStreamReader("data/data4cyber_run1.csv")
   for record in reader.stream_records():
       print(record)
   ```
