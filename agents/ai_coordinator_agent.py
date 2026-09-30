"""
agents/ai_coordinator_agent.py
================================
Central AI Agent powered by Ollama (local LLM).

This agent receives reports from all 4 specialist agents
(Solar, Demand, Battery, Price) + current telemetry and
generates:
  1. Natural-language strategic reasoning explaining WHY the 
     current action was chosen
  2. 4-hour ahead scenario prediction
  3. Risk assessment and recommendation
  4. Anomaly/alert narrative

Falls back to a rule-based summary if Ollama is unavailable.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional
from urllib import request, error


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"
TIMEOUT_SEC = 90  # streaming mode: 50 tokens on CPU takes ~30-60s

SYSTEM_PROMPT = "Energy AI: Be concise. End with RECOMMEND: one action."


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> Optional[str]:
    """Call local Ollama API using STREAMING mode for CPU-tolerance.
    
    stream=True sends back one JSON line per token immediately,
    so we never hit a single-response timeout on slow CPU hardware.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,          # stream tokens as they generate
        "options": {
            "temperature": 0.2,
            "num_predict": 50,   # strict 50-token max for CPU speed (~15-25s)
            "top_k": 20,
            "top_p": 0.9,
        }
    }).encode()
    try:
        import socket as _socket
        req = request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        tokens: list[str] = []
        deadline = time.time() + TIMEOUT_SEC
        with request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            for raw_line in resp:
                if time.time() > deadline:
                    break
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                chunk = json.loads(raw_line.decode())
                tokens.append(chunk.get("response", ""))
                if chunk.get("done", False):
                    break
        return "".join(tokens).strip() or None
    except Exception:
        return None



def _fallback_reasoning(
    state: Dict[str, Any],
    solar_rep: Dict[str, Any],
    demand_rep: Dict[str, Any],
    battery_rep: Dict[str, Any],
    price_rep: Dict[str, Any],
    action_name: str,
    safety_status: str
) -> str:
    """Rule-based fallback reasoning when LLM is offline."""
    solar = float(state.get("solar_kw", 0))
    load = float(state.get("load_kw", 0))
    soc = float(state.get("soc", 0.5)) * 100
    price = float(state.get("price", 7.0))
    tier = price_rep.get("tariff_tier", "NORMAL")
    surplus = solar_rep.get("current_surplus_kw", 0)
    batt_health = battery_rep.get("health_flag", "HEALTHY")

    lines = [
        f"Solar: {solar:.1f}kW | Load: {load:.1f}kW | SoC: {soc:.0f}% | Tariff: {tier}",
        f"Action dispatched: {action_name} | Safety: {safety_status}",
    ]
    if surplus > 0.5:
        lines.append(f"Surplus solar {surplus:.1f}kW available — battery charging optimal.")
    elif soc < 25:
        lines.append("Battery critically low — grid import or load shedding advisable.")
    if tier == "PEAK_EXPENSIVE":
        lines.append("Peak tariff active — maximize battery discharge, minimize grid import.")
    if batt_health != "HEALTHY":
        lines.append(f"WARNING: Battery health flag = {batt_health}. Reduce C-rate.")
    lines.append("RECOMMEND: Maintain current strategy and re-evaluate at next 15-min interval.")
    return " ".join(lines)


def generate_ai_reasoning(
    state: Dict[str, Any],
    solar_rep: Dict[str, Any],
    demand_rep: Dict[str, Any],
    battery_rep: Dict[str, Any],
    price_rep: Dict[str, Any],
    action_name: str,
    safety_status: str,
) -> Dict[str, Any]:
    """
    Main entry point: generate AI reasoning for a single control tick.

    Returns a dict with:
      - reasoning: str    (LLM or fallback narrative)
      - source: str       ('ollama' | 'fallback')
      - model: str
      - latency_ms: float
    """
    t0 = time.time()

    # Build rich telemetry prompt
    solar = float(state.get("solar_kw", 0))
    load = float(state.get("load_kw", 0))
    soc = float(state.get("soc", 0.5)) * 100
    price = float(state.get("price", 7.0))
    hour = float(state.get("hour", 12))
    solar_fc_1h = float(state.get("solar_fc_1h", solar))
    load_fc_1h = float(state.get("load_fc_1h", load))
    solar_fc_4h = float(state.get("solar_fc_4h", solar_fc_1h))
    load_fc_4h = float(state.get("load_fc_4h", load_fc_1h))
    surplus = solar_rep.get("current_surplus_kw", 0)
    tier = price_rep.get("tariff_tier", "NORMAL")
    batt_health = battery_rep.get("health_flag", "HEALTHY")
    peak_risk = demand_rep.get("peak_risk", False)
    avail_charge = battery_rep.get("avail_charge_kw", 0)
    avail_discharge = battery_rep.get("avail_discharge_kw", 0)

    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"Solar={solar:.1f}kW Load={load:.1f}kW SoC={soc:.0f}% "
        f"Tariff={tier}(Rs.{price:.1f}) Action={action_name} Safety={safety_status} "
        f"Forecast4h: solar={solar_fc_4h:.1f}kW load={load_fc_4h:.1f}kW. "
        f"Analyse and RECOMMEND:"
    )


    llm_text = _call_ollama(prompt)
    latency_ms = (time.time() - t0) * 1000

    if llm_text:
        source = "ollama"
        reasoning = llm_text
    else:
        source = "fallback"
        reasoning = _fallback_reasoning(
            state, solar_rep, demand_rep, battery_rep, price_rep,
            action_name, safety_status
        )

    return {
        "reasoning": reasoning,
        "source": source,
        "model": OLLAMA_MODEL if source == "ollama" else "rule_based",
        "latency_ms": round(latency_ms, 1),
    }
