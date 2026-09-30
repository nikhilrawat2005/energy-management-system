import csv
from coordinator.coordinator import MultiAgentCoordinator

def run_evaluation_comparison(dataset_path="data/dataset_15min.csv", days_to_eval=7):
    """
    Evaluates:
      1. Grid-Only Baseline (no solar storage, battery idle)
      2. Rule-Based Multi-Agent Coordinator
      3. RL Policy Multi-Agent Coordinator
    """
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    eval_steps = min(len(rows), days_to_eval * 96)
    rows = rows[:eval_steps]
    dt = 0.25 # 15 min
    cap_kwh = 15.0
    pmax_kw = 5.0
    eff = 0.95
    
    def simulate_strategy(mode):
        coord = MultiAgentCoordinator()
        soc = 0.5
        total_grid_cost = 0.0
        total_grid_import_kwh = 0.0
        total_solar_gen_kwh = 0.0
        total_solar_used_kwh = 0.0
        total_curtailed_kwh = 0.0
        total_load_kwh = 0.0
        battery_throughput_kwh = 0.0
        safety_overrides = 0
        
        for i, r in enumerate(rows):
            state = dict(r)
            state["soc"] = soc
            solar = float(r["solar_kw"])
            load = float(r["load_kw"])
            price = float(r["price"])
            grid_status = int(r["grid_status"])
            
            total_solar_gen_kwh += solar * dt
            total_load_kwh += load * dt
            
            if mode == "grid_only":
                action = 0
            else:
                out = coord.process_tick(state, mode=mode, current_time_sec=i * 900.0)
                action = out["final_action"]
                if "OVERRIDE" in out["safety_status"]:
                    safety_overrides += 1
                    
            batt_kw = 0.0
            if action == 1: # charge solar
                surplus = max(0.0, solar - load)
                can_chg = (0.95 - soc) * cap_kwh / dt
                batt_kw = min(pmax_kw, surplus, can_chg)
            elif action == 2: # discharge
                deficit = max(0.0, load - solar)
                can_dis = (soc - 0.10) * cap_kwh / dt
                batt_kw = -min(pmax_kw, deficit, can_dis)
            elif action == 3 and grid_status == 1: # charge grid
                can_chg = (0.95 - soc) * cap_kwh / dt
                batt_kw = min(pmax_kw, can_chg)
                
            # Update SoC
            if batt_kw > 0:
                soc += (batt_kw * eff * dt) / cap_kwh
            elif batt_kw < 0:
                soc += (batt_kw / eff * dt) / cap_kwh
            soc = min(max(soc, 0.10), 0.95)
            battery_throughput_kwh += abs(batt_kw) * dt
            
            # Flow balance
            solar_to_load = min(solar, load)
            if batt_kw > 0:
                used = solar_to_load + batt_kw
                curtailed = max(0.0, solar - used)
                shortfall = max(0.0, load - solar_to_load)
                g_import = shortfall if grid_status == 1 else 0.0
                if action == 3:
                    g_import += batt_kw
            else:
                batt_dis = abs(batt_kw)
                shortfall = max(0.0, load - (solar_to_load + batt_dis))
                g_import = shortfall if grid_status == 1 else 0.0
                curtailed = max(0.0, solar - solar_to_load)
                used = solar_to_load
                
            total_grid_cost += g_import * price * dt
            total_grid_import_kwh += g_import * dt
            total_solar_used_kwh += used * dt
            total_curtailed_kwh += curtailed * dt

        renewable_util = (total_solar_used_kwh / max(1e-5, total_solar_gen_kwh)) * 100.0
        grid_dep = (total_grid_import_kwh / max(1e-5, total_load_kwh)) * 100.0
        batt_cycles = battery_throughput_kwh / (2.0 * cap_kwh)
        
        return {
            "mode": mode,
            "grid_cost_inr": round(total_grid_cost, 2),
            "solar_used_kwh": round(total_solar_used_kwh, 2),
            "curtailed_solar_kwh": round(total_curtailed_kwh, 2),
            "renewable_utilization_pct": round(renewable_util, 2),
            "grid_dependency_pct": round(grid_dep, 2),
            "battery_cycles": round(batt_cycles, 2),
            "safety_overrides": safety_overrides
        }

    res_grid = simulate_strategy("grid_only")
    res_rules = simulate_strategy("rule_based")
    res_rl = simulate_strategy("rl")

    savings_rules = res_grid['grid_cost_inr'] - res_rules['grid_cost_inr']
    savings_rl = res_grid['grid_cost_inr'] - res_rl['grid_cost_inr']
    pct_save_rules = (savings_rules / res_grid['grid_cost_inr']) * 100.0
    pct_save_rl = (savings_rl / res_grid['grid_cost_inr']) * 100.0

    print("\n==================== 7-DAY BENCHMARK COMPARISON ====================")
    print(f"{'Metric':<30} | {'Grid-Only':<15} | {'Rule-Based':<15} | {'RL Agent':<15}")
    print("-" * 80)
    print(f"{'Electricity Cost (INR)':<30} | Rs.{res_grid['grid_cost_inr']:<13} | Rs.{res_rules['grid_cost_inr']:<13} | Rs.{res_rl['grid_cost_inr']:<13}")
    print(f"{'Savings vs Grid-Only':<30} | {'0.0%':<15} | Rs.{savings_rules:.1f} ({pct_save_rules:.1f}%) | Rs.{savings_rl:.1f} ({pct_save_rl:.1f}%)")
    print(f"{'Renewable Utilization (%)':<30} | {res_grid['renewable_utilization_pct']}%         | {res_rules['renewable_utilization_pct']}%         | {res_rl['renewable_utilization_pct']}%")
    print(f"{'Wasted Solar (kWh)':<30} | {res_grid['curtailed_solar_kwh']:<15} | {res_rules['curtailed_solar_kwh']:<15} | {res_rl['curtailed_solar_kwh']:<15}")
    print(f"{'Grid Dependency (%)':<30} | {res_grid['grid_dependency_pct']}%         | {res_rules['grid_dependency_pct']}%         | {res_rl['grid_dependency_pct']}%")
    print(f"{'Battery Equivalent Cycles':<30} | {res_grid['battery_cycles']:<15} | {res_rules['battery_cycles']:<15} | {res_rl['battery_cycles']:<15}")
    print(f"{'Safety Overrides Count':<30} | {res_grid['safety_overrides']:<15} | {res_rules['safety_overrides']:<15} | {res_rl['safety_overrides']:<15}")
    print("=====================================================================\n")

if __name__ == "__main__":
    run_evaluation_comparison()
