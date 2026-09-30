"""7-day strategy benchmark: grid-only vs rule-based vs learned RL policy.

Bugs fixed
----------
1. **Duplicated physics.** The old version re-implemented the power balance,
   SoC integration and tariff arithmetic that already exist in
   :class:`coordinator.rl_env.EnergyEnv`, and hard-coded ``0.25 h`` /
   ``15 kWh`` / ``5 kW`` / ``0.95`` / ``0.10``/``0.95`` instead of reading
   ``config/limits.yaml``. Any retune of the battery envelope silently
   desynchronised the benchmark from the thing it was measuring. The rollout
   below now drives the *same* :class:`EnergyEnv` the policy was trained on, so
   training and evaluation can no longer disagree.
2. **Relative dataset path.** ``"data/dataset_15min.csv"`` only resolved when
   the process happened to be started from the project root.
3. **ZeroDivisionError.** ``savings / res_grid['grid_cost_inr']`` blew up on a
   window with no grid import at all (for example a fully solar-covered day,
   or an all-blackout slice).
4. **Unused coordinator for the grid-only arm.** Every arm constructed a
   :class:`MultiAgentCoordinator`, which loads config *and* the RL policy from
   disk; the grid-only arm never uses either.
5. **Safety-override detection by substring.** ``"OVERRIDE" in
   out["safety_status"]`` missed the ``SAFETY_ISLAND_MODE`` and
   ``SAFETY_RATE_LIMIT`` statuses. The result is now taken from the explicit
   ``safety_override`` boolean the coordinator returns.
6. **No unmet-load or action-distribution reporting**, so a strategy could look
   cheap purely by shedding load during outages.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Dict, List, Optional

import project_paths
from coordinator.coordinator import MultiAgentCoordinator
from coordinator.rl_env import EnergyEnv

# One arm's totals over the evaluation window.
StrategyResult = Dict[str, Any]


def _pct(numerator: float, denominator: float) -> float:
    """Percentage that degrades to 0.0 instead of raising on a zero baseline."""
    if denominator <= 0.0:
        return 0.0
    return (numerator / denominator) * 100.0


def _action_name(action_id: int) -> str:
    from coordinator.coordinator import action_name

    return action_name(action_id)


def run_strategy(
    mode: str,
    env: EnergyEnv,
    eval_steps: int,
    coordinator: Optional[MultiAgentCoordinator] = None,
) -> StrategyResult:
    """Roll one strategy over the first *eval_steps* samples of the dataset.

    The environment owns the physics; ``mode`` only decides which action is
    requested each step. ``coordinator`` is built lazily so the grid-only arm
    does not pay for loading config and the RL policy from disk.
    """
    if mode not in ("grid_only", "rule_based", "rl"):
        raise ValueError(f"Unknown strategy mode: {mode!r}")

    obs = env.reset()
    done = False
    steps = 0

    totals: Dict[str, float] = {
        "grid_cost_inr": 0.0,
        "grid_import_kwh": 0.0,
        "solar_gen_kwh": 0.0,
        "solar_used_kwh": 0.0,
        "curtailed_solar_kwh": 0.0,
        "load_kwh": 0.0,
        "battery_throughput_kwh": 0.0,
        "unmet_load_kwh": 0.0,
    }
    action_counts: Dict[str, int] = {}
    safety_overrides = 0
    trip_reasons: Dict[str, int] = {}
    control_interval_sec = env.dt * 3600.0

    if coordinator is None:
        coordinator = MultiAgentCoordinator()
    coordinator.reset()

    while not done and steps < eval_steps:
        solar = float(env.rows[env.i]["solar_kw"])
        load = float(env.rows[env.i]["load_kw"])
        totals["solar_gen_kwh"] += solar * env.dt
        totals["load_kwh"] += load * env.dt

        if mode == "grid_only":
            action = 0
        else:
            state = dict(env.rows[env.i])
            state["soc"] = env.soc
            outcome = coordinator.process_tick(
                state, mode=mode, current_time_sec=steps * control_interval_sec
            )
            action = int(outcome["final_action"])
            if outcome.get("safety_override"):
                safety_overrides += 1
                reason = str(outcome.get("safety_status", "")).split(":", 1)[0]
                trip_reasons[reason] = trip_reasons.get(reason, 0) + 1

        obs, _reward, done, info = env.step(action)
        steps += 1

        action_counts[_action_name(action)] = action_counts.get(_action_name(action), 0) + 1
        totals["grid_cost_inr"] += info["grid_cost_inr"]
        totals["grid_import_kwh"] += info["grid_import_kwh"]
        totals["solar_used_kwh"] += info["solar_used_kwh"]
        totals["curtailed_solar_kwh"] += info["curtailed_solar_kwh"]
        totals["unmet_load_kwh"] += info["unmet_load_kwh"]
        totals["battery_throughput_kwh"] += abs(info["batt_kw"]) * env.dt

    return {
        "mode": mode,
        "steps": steps,
        "grid_cost_inr": round(totals["grid_cost_inr"], 2),
        "grid_import_kwh": round(totals["grid_import_kwh"], 2),
        "solar_gen_kwh": round(totals["solar_gen_kwh"], 2),
        "solar_used_kwh": round(totals["solar_used_kwh"], 2),
        "curtailed_solar_kwh": round(totals["curtailed_solar_kwh"], 2),
        "load_kwh": round(totals["load_kwh"], 2),
        "unmet_load_kwh": round(totals["unmet_load_kwh"], 2),
        "renewable_utilization_pct": round(
            _pct(totals["solar_used_kwh"], totals["solar_gen_kwh"]), 2
        ),
        "grid_dependency_pct": round(
            _pct(totals["grid_import_kwh"], totals["load_kwh"]), 2
        ),
        "battery_cycles": round(totals["battery_throughput_kwh"] / (2.0 * env.cap_kwh), 2),
        "safety_overrides": safety_overrides,
        "safety_trip_reasons": trip_reasons,
        "action_distribution": action_counts,
    }


def run_evaluation_comparison(
    dataset_path: Optional[str] = None,
    days_to_eval: int = 7,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate grid-only, rule-based and RL strategies on a common window.

    Returns a dict with each strategy's totals plus the derived savings, so the
    function is usable from tests instead of only printing a table.
    """
    if days_to_eval <= 0:
        raise ValueError("days_to_eval must be a positive integer")

    env = EnergyEnv(dataset_path or project_paths.DEFAULT_DATASET_PATH)
    eval_steps = min(env.n_steps, int(days_to_eval * 24 * 3600 / env.dt / 3600))

    # One coordinator is shared by the two agent-driven arms so the safety
    # layer's rate limiter sees a continuous timeline across the window.
    coordinator = MultiAgentCoordinator()

    results: Dict[str, StrategyResult] = {
        "grid_only": run_strategy("grid_only", env, eval_steps, coordinator),
        "rule_based": run_strategy("rule_based", env, eval_steps, coordinator),
        "rl": run_strategy("rl", env, eval_steps, coordinator),
    }

    grid_cost = results["grid_only"]["grid_cost_inr"]
    for key, res in results.items():
        res["savings_inr"] = round(grid_cost - res["grid_cost_inr"], 2)
        res["savings_pct"] = round(_pct(grid_cost - res["grid_cost_inr"], grid_cost), 2)

    if verbose:
        _print_report(results, days_to_eval, eval_steps, env)

    return {"window": {"days": days_to_eval, "steps": eval_steps, "dataset": env.data_path},
            "results": results}


def _print_report(
    results: Dict[str, StrategyResult],
    days_to_eval: int,
    eval_steps: int,
    env: EnergyEnv,
) -> None:
    labels = {"grid_only": "Grid-Only", "rule_based": "Rule-Based", "rl": "RL Agent"}
    width = 15
    header = f"{'Metric':<32} |" + "".join(
        f" {labels[k]:<{width - 1}} |" for k in ("grid_only", "rule_based", "rl")
    )

    print("\n" + "=" * len(header))
    print(f" {days_to_eval}-DAY BENCHMARK COMPARISON "
          f"({eval_steps} steps @ {env.dt * 60:.0f} min, capacity {env.cap_kwh:g} kWh)")
    print(header)
    print("-" * len(header))

    def row(label: str, key: str, prefix: str = "", suffix: str = "") -> None:
        cells = "".join(
            f" {prefix}{results[k][key]}{suffix:<{max(0, width - 1 - len(prefix) - len(str(results[k][key])))}} |"
            for k in ("grid_only", "rule_based", "rl")
        )
        print(f"{label:<32} |{cells}")

    row("Electricity Cost (INR)", "grid_cost_inr", "Rs.")
    row("Savings vs Grid-Only (INR)", "savings_inr", "Rs.")
    row("Savings vs Grid-Only (%)", "savings_pct", "", "%")
    row("Grid Import (kWh)", "grid_import_kwh")
    row("Grid Dependency (%)", "grid_dependency_pct", "", "%")
    row("Renewable Utilization (%)", "renewable_utilization_pct", "", "%")
    row("Curtailed Solar (kWh)", "curtailed_solar_kwh")
    row("Unmet Load (kWh)", "unmet_load_kwh")
    row("Battery Equivalent Cycles", "battery_cycles")
    row("Safety Interlocks Tripped", "safety_overrides")
    print("-" * len(header))

    for key in ("rule_based", "rl"):
        dist = results[key]["action_distribution"]
        trips = results[key]["safety_trip_reasons"]
        print(f"  {labels[key]} actions: " + ", ".join(f"{a}={n}" for a, n in sorted(dist.items())))
        print(f"  {labels[key]} interlock reasons: " + (", ".join(
            f"{r}={n}" for r, n in sorted(trips.items())) if trips else "none"))

    print("=" * len(header) + "\n")


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="MAEMS strategy benchmark")
    parser.add_argument("--days", type=int, default=7, help="evaluation window in days")
    parser.add_argument("--dataset", default=None, help="dataset CSV (default: data/dataset_15min.csv)")
    args = parser.parse_args(argv)
    run_evaluation_comparison(dataset_path=args.dataset, days_to_eval=args.days)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
