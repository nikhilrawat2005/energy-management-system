import csv
from coordinator.coordinator import MultiAgentCoordinator

def test_data4cyber_run():
    coord = MultiAgentCoordinator()
    rows = []
    with open("data/data4cyber_maems_processed.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Running Multi-Agent System on {len(rows)} real Data4Cyber telemetry steps...\n")

    actions_taken = {0: 0, 1: 0, 2: 0, 3: 0}
    for i, r in enumerate(rows):
        res = coord.process_tick(r, mode="rule_based", current_time_sec=i * 5.0)
        a = res["final_action"]
        actions_taken[a] += 1
        if i < 6 or i == len(rows) - 1:
            print(f"[Step {i+1}] Solar: {r['solar_kw']} kW | Load: {r['load_kw']} kW | SoC: {float(r['soc'])*100:.1f}% -> Decision: {res['final_action_name']}")

    print("\nAction Distribution across Data4Cyber records:")
    for a, cnt in actions_taken.items():
        print(f"  Action {a}: {cnt} steps")

if __name__ == "__main__":
    test_data4cyber_run()
