"""Single-entrypoint launcher for the Multi-Agent Energy Management System (MAEMS).

Starts:
1. Docker Stack (TimescaleDB, Mosquitto MQTT, Grafana Dashboard, Ollama AI)
2. FastAPI Service (control endpoints, Swagger UI at http://127.0.0.1:8000/docs)
3. Ollama Model Warmup (pre-loads qwen2.5:1.5b into GPU/CPU memory)
4. Live Streamer (real-time stream to TimescaleDB & Grafana dashboard)
"""

import os
import sys
import time
import subprocess
import threading
import uvicorn

# Ensure root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.main import app
from ingestion.live_streamer import run_unified_streamer

OLLAMA_MODEL = "qwen2.5:1.5b"


def check_and_start_docker():
    """Ensure docker containers are running."""
    try:
        subprocess.run(["docker", "compose", "up", "-d"], cwd=_PROJECT_ROOT, check=False)
    except Exception as e:
        print(f"[Docker] Notice: Could not invoke docker compose automatically: {e}")


def warmup_ollama():
    """Pre-load the Ollama model so first real query is fast."""
    try:
        import urllib.request, json
        print(f"[Ollama] Warming up {OLLAMA_MODEL}...")
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": "Summarize: solar=5kW, load=3kW, soc=70%, action=CHARGE_SOLAR",
            "stream": False,
            "options": {"num_predict": 30}
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        print(f"[Ollama] Model {OLLAMA_MODEL} is warm and ready!")
    except Exception as e:
        print(f"[Ollama] Warmup skipped (will lazy-load on first query): {e}")


def run_api_server():
    """Runs FastAPI backend using Uvicorn."""
    print("[API] Starting MAEMS FastAPI service on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


def main():
    print("=" * 80)
    print("      MULTI-AGENT ENERGY MANAGEMENT SYSTEM (MAEMS)")
    print("=" * 80)
    print(">> Initializing Docker services (TimescaleDB + Mosquitto + Grafana + Ollama)...")
    check_and_start_docker()

    # Start API in background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()

    # Warmup Ollama in background (non-blocking)
    ollama_thread = threading.Thread(target=warmup_ollama, daemon=True)
    ollama_thread.start()

    time.sleep(1.5)
    print("\n" + "=" * 80)
    print("   SERVICES ACTIVE:")
    print("   - Grafana Dashboard:   http://localhost:3001  (User: admin / Pass: admin)")
    print("   - FastAPI Swagger UI:  http://127.0.0.1:8000/docs")
    print("   - TimescaleDB Port:    5434")
    print("   - Mosquitto MQTT:      1883")
    print("   - Ollama AI Port:      11434  (qwen2.5:1.5b warming up...)")
    print("   - Live Telemetry:      Streaming live into TimescaleDB & Dashboards")
    print("   - AI Reasoning:        Every 5 ticks (10s wall-clock)")
    print("   - Press Ctrl+C anytime to stop")
    print("=" * 80 + "\n")

    # Run live streamer continuously to feed Grafana in real-time
    try:
        run_unified_streamer(wall_interval_sec=2.0, mode="rl", verbose=True)
    except KeyboardInterrupt:
        print("\n[MAEMS] System stopped by user. Goodbye!")


if __name__ == "__main__":
    main()
