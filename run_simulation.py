import time
import csv
from coordinator.coordinator import MultiAgentCoordinator

def run_live_simulation(num_steps=12, interval_sec=1.0):
    print("=" * 80)
    print("      MULTI-AGENT ENERGY MANAGEMENT SYSTEM (MAEMS) - LIVE SIMULATOR")
    print("=" * 80)
    
    coord = MultiAgentCoordinator()
    
    rows = []
    with open("data/dataset_15min.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    # Sample afternoon peak solar to evening transition steps (rows 40 to 60)
    selected_rows = rows[40:40 + num_steps]
    
    soc = 0.40 # starting 40% SoC
    
    for idx, r in enumerate(selected_rows):
        state = dict(r)
        state["soc"] = soc
        
        # Dispatch through coordinator
        res = coord.process_tick(state, mode="rule_based", current_time_sec=time.time())
        
        solar = float(state["solar_kw"])
        load = float(state["load_kw"])
        hour = float(state["hour"])
        price = float(state["price"])
        
        solar_rep = res["agent_reports"]["solar"]
        demand_rep = res["agent_reports"]["demand"]
        battery_rep = res["agent_reports"]["battery"]
        price_rep = res["agent_reports"]["price"]
        
        print(f"\n[Time: {int(hour):02d}:{int((hour%1)*60):02d} | Step {idx+1}/{num_steps}]")
        print(f"  Sensors -> Solar: {solar:4.2f} kW | Load: {load:4.2f} kW | Price: Rs.{price:.1f}/kWh | SoC: {soc*100:.1f}%")
        print(f"  Agents  -> Solar Surplus: {solar_rep['current_surplus_kw']} kW | Peak Risk: {demand_rep['peak_risk']} | Batt Health: {battery_rep['health_flag']}")
        print(f"  Decision-> Action: {res['final_action_name']}")
        print(f"  Safety  -> {res['safety_status']}")
        
        # Simple battery state step
        action = res["final_action"]
        if action == 1:
            soc = min(0.95, soc + 0.06)
        elif action == 2:
            soc = max(0.10, soc - 0.05)
            
        time.sleep(interval_sec)
        
    print("\n" + "=" * 80)
    print("Simulation run completed successfully.")
    print("=" * 80)

if __name__ == "__main__":
    run_live_simulation(num_steps=6, interval_sec=0.2)
