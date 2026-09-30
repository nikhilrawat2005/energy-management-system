import numpy as np
import json
import os
from coordinator.rl_env import EnergyEnv

class QLearningEnergyPolicy:
    """
    Compact Tabular / Tile-coded RL agent for fast, robust policy optimization
    without heavy C-runtime external dependencies.
    Optimizes for peak-shaving, solar self-consumption, and lowest grid cost.
    """
    def __init__(self, n_actions=4, lr=0.1, gamma=0.98, epsilon=0.15):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def _discretize_state(self, obs: np.ndarray) -> str:
        # State: [solar, load, soc, batt_temp, price, solar_fc_1h, load_fc_1h, sin_h, cos_h]
        solar_bin = int(min(obs[0] // 2.0, 4))
        load_bin = int(min(obs[1] // 2.5, 4))
        soc_bin = int(min(obs[2] // 0.20, 4))
        price_bin = 0 if obs[4] < 6.0 else (2 if obs[4] > 10.0 else 1)
        surplus_bin = 1 if (obs[0] > obs[1]) else 0
        return f"{solar_bin}_{load_bin}_{soc_bin}_{price_bin}_{surplus_bin}"

    def get_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        state_key = self._discretize_state(obs)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.n_actions
            
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state_key]))

    def update(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        state_key = self._discretize_state(obs)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.n_actions
            
        next_key = self._discretize_state(next_obs)
        if next_key not in self.q_table:
            self.q_table[next_key] = [0.0] * self.n_actions
            
        best_next_q = 0.0 if done else max(self.q_table[next_key])
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * (reward + self.gamma * best_next_q - current_q)

    def train(self, env: EnergyEnv, episodes: int = 15, model_save_path: str = "models/rl_policy.json"):
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        print(f"[RL Policy] Starting training for {episodes} episodes over environment...")
        
        for ep in range(episodes):
            obs = env.reset()
            total_reward = 0.0
            done = False
            
            while not done:
                action = self.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                self.update(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward
                
            self.epsilon = max(0.01, self.epsilon * 0.90)
            if (ep + 1) % 5 == 0 or ep == episodes - 1:
                print(f" Episode {ep + 1}/{episodes} completed. Total Cumulative Reward: {total_reward:.2f}")

        with open(model_save_path, "w", encoding="utf-8") as f:
            json.dump(self.q_table, f)
        print(f"[RL Policy] Model saved to {model_save_path}")

    def load(self, model_save_path: str = "models/rl_policy.json"):
        if os.path.exists(model_save_path):
            with open(model_save_path, "r", encoding="utf-8") as f:
                self.q_table = json.load(f)
            print(f"[RL Policy] Loaded policy model with {len(self.q_table)} state buckets.")
        else:
            print("[RL Policy] No saved model found. Initialized fresh.")

if __name__ == "__main__":
    env = EnergyEnv("data/dataset_15min.csv")
    agent = QLearningEnergyPolicy()
    agent.train(env, episodes=20)
