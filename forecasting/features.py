import numpy as np
import csv
import math
import json
import os
from typing import List, Dict, Tuple

class SimpleGradientBoostingRegressor:
    """
    Lightweight, fast Gradient Boosted Decision Trees Regressor implemented
    in pure Python/NumPy (100% offline, zero C-DLL dependency issues).
    Uses shallow regression trees to predict continuous targets (kW load or solar).
    """
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.08, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.base_pred = 0.0
        self.trees = []

    class DecisionTree:
        def __init__(self, max_depth=3):
            self.max_depth = max_depth
            self.tree = None

        def _split(self, X, y, depth):
            if depth >= self.max_depth or len(y) <= 5:
                return {"val": float(np.mean(y))}

            best_feat, best_val, best_mse = None, None, float("inf")
            n_samples, n_feats = X.shape

            # Subsample features for speed
            for feat in range(n_feats):
                vals = np.percentile(X[:, feat], [20, 40, 60, 80])
                for v in vals:
                    left_mask = X[:, feat] <= v
                    right_mask = ~left_mask
                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue
                    y_l, y_r = y[left_mask], y[right_mask]
                    mse = np.sum((y_l - np.mean(y_l))**2) + np.sum((y_r - np.mean(y_r))**2)
                    if mse < best_mse:
                        best_mse = mse
                        best_feat = feat
                        best_val = v

            if best_feat is None:
                return {"val": float(np.mean(y))}

            left_mask = X[:, best_feat] <= best_val
            return {
                "feat": best_feat,
                "split": best_val,
                "left": self._split(X[left_mask], y[left_mask], depth + 1),
                "right": self._split(X[~left_mask], y[~left_mask], depth + 1)
            }

        def fit(self, X, y):
            self.tree = self._split(X, y, 0)

        def _predict_row(self, node, row):
            if "val" in node:
                return node["val"]
            if row[node["feat"]] <= node["split"]:
                return self._predict_row(node["left"], row)
            return self._predict_row(node["right"], row)

        def predict(self, X):
            return np.array([self._predict_row(self.tree, row) for row in X])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_pred = float(np.mean(y))
        y_curr = np.full(len(y), self.base_pred)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - y_curr
            tree = self.DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            preds = tree.predict(X)
            y_curr += self.lr * preds
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        res = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            res += self.lr * tree.predict(X)
        return res

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Serialize tree structure to json
        def serialize_node(node):
            if "val" in node:
                return {"val": node["val"]}
            return {
                "feat": int(node["feat"]),
                "split": float(node["split"]),
                "left": serialize_node(node["left"]),
                "right": serialize_node(node["right"])
            }
        serialized = {
            "base_pred": self.base_pred,
            "lr": self.lr,
            "trees": [serialize_node(t.tree) for t in self.trees]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.base_pred = data["base_pred"]
        self.lr = data["lr"]
        self.trees = []
        for t_dict in data["trees"]:
            tree = self.DecisionTree()
            tree.tree = t_dict
            self.trees.append(tree)
