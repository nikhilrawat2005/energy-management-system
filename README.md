# Multi-Agent Energy Management System (MAEMS)

Real-time data decoding, forecasting, specialized agents (Solar, Demand, Battery, Price), safety interlock, and an energy coordinator (Rule-Based + Reinforcement Learning).

## Architecture & Features
- **Sensor Ingestion & Modbus Decoding**: Modbus RTU/TCP float32/uint32 IEEE-754 register decoding, physical boundary filtering, stuck-sensor detection.
- **Specialized Agents**:
  - `SolarAgent`: Monitors PV output, irradiance, and expected horizon surplus.
  - `DemandAgent`: Monitors consumption and detects peak demand risk.
  - `BatteryAgent`: Tracks SoC (10%-95%), battery thermal status (45°C/55°C limit), and enforces dynamic C-rates.
  - `PriceAgent`: Time-of-Day (ToD) tariff optimization and grid status tracking.
- **Safety Interlock Layer**: Critical thermal lockout, overcharge/overdischarge protection, and grid-outage microgrid islanding.
- **Coordinator**:
  - Baseline Rule-based heuristic coordinator.
  - Q-Learning / RL Policy trained for solar self-consumption and peak cost reduction.
- **Evaluation & Benchmarking**: 7-day comparative analysis across Grid-Only, Rule-Based, and RL strategies.
- **FastAPI Service**: Live endpoints for control dispatch, agent status, and limits.

## Quick Start

### 1. Run Benchmark Evaluation
```bash
python evaluate.py
```

### 2. Run Interactive Simulation Demo
```bash
python run_simulation.py
```

### 3. Start FastAPI Service
```bash
python -m uvicorn api.main:app --reload
```
API Documentation available at: `http://127.0.0.1:8000/docs`
