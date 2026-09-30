"""
Single source of truth for every filesystem path and YAML config used by MAEMS.

Why this module exists
----------------------
Historically every module opened files with hard-coded *relative* paths such as
``"config/limits.yaml"`` or ``"models/rl_policy.json"``. Those only resolve when
the process happens to be started from the project root, so the same code broke
under ``uvicorn``, ``python api/main.py``, systemd, Docker, or any test runner
invoked from another directory.

Every path in the project is now resolved against ``PROJECT_ROOT`` (the parent
directory of this file) so behaviour is identical no matter the working
directory. Callers may still pass absolute paths, and relative paths are treated
as project-relative.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))

CONFIG_DIR: str = os.path.join(PROJECT_ROOT, "config")
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")
GRAFANA_DASHBOARD_DIR: str = os.path.join(
    CONFIG_DIR, "grafana", "provisioning", "dashboards"
)

LIMITS_PATH: str = os.path.join(CONFIG_DIR, "limits.yaml")
DEVICES_PATH: str = os.path.join(CONFIG_DIR, "devices.yaml")

RL_POLICY_PATH: str = os.path.join(MODELS_DIR, "rl_policy.json")
DEMAND_MODEL_PATH: str = os.path.join(MODELS_DIR, "demand_forecast.json")
SOLAR_MODEL_PATH: str = os.path.join(MODELS_DIR, "solar_forecast.json")

DEFAULT_DATASET_PATH: str = os.path.join(DATA_DIR, "dataset_15min.csv")
DATA4CYBER_DATASET_PATH: str = os.path.join(DATA_DIR, "data4cyber_maems_processed.csv")
DATA4CYBER_RAW_DATASET_PATH: str = os.path.join(
    DATA_DIR, "raw_data4cyber", "data4cyber_dataset", "S1_industroyer_pv", "dataset.csv"
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def resolve(path: str) -> str:
    """Return an absolute path for *path*.

    Absolute paths are returned untouched (normalised). Relative paths are
    interpreted relative to the project root, which is what every module in
    this codebase means by e.g. ``"config/limits.yaml"``.
    """
    if path is None:
        raise ValueError("path must not be None")
    path = str(path)
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def ensure_parent_dir(path: str) -> str:
    """Create the parent directory of *path* if needed and return the path.

    Safe for bare filenames (``ensure_parent_dir("out.json")``) where
    ``os.path.dirname`` returns an empty string and ``os.makedirs("")`` raises
    ``FileNotFoundError``.
    """
    resolved = resolve(path)
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return resolved


def ensure_log_dir() -> str:
    """Create and return the ``logs/`` directory used by the PowerShell runner."""
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR


# ---------------------------------------------------------------------------
# YAML config helpers
# ---------------------------------------------------------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML mapping, raising a clear error if the file is missing/empty."""
    resolved = resolve(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Required YAML config not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}, got {type(data).__name__}")
    return data


def load_limits(path: Optional[str] = None) -> Dict[str, Any]:
    """Load ``config/limits.yaml`` (battery / grid / safety / tariff / demand)."""
    return load_yaml(path or LIMITS_PATH)


def load_devices(path: Optional[str] = None) -> Dict[str, Any]:
    """Load ``config/devices.yaml`` (Modbus register map + sensor sanity ranges).

    Previously this file shipped in the repo but was never read by any module,
    so the sanitisation limits in it had no effect. The Modbus reader now loads
    it through this helper.
    """
    return load_yaml(path or DEVICES_PATH)


def battery_section(limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the ``battery:`` block of the limits config, never ``None``."""
    return dict((limits or load_limits()).get("battery", {}) or {})


def control_interval_hours(limits: Optional[Dict[str, Any]] = None) -> float:
    """Control/dispatch period in hours, read from config (default 15 min).

    The dispatch period is the single time base used for SoC integration,
    energy metering, cost accounting and battery wear. Previously several
    modules hard-coded ``0.25`` while the live streamer integrated energy over
    its 2-second wall-clock sleep, so the reported savings and meter registers
    disagreed with the physics by a factor of ~450.
    """
    cfg = limits or load_limits()
    seconds = float(cfg.get("safety", {}).get("rate_limit_seconds", 900) or 900)
    return seconds / 3600.0
