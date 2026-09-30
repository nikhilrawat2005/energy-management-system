import csv
import numpy as np
import os
from forecasting.features import SimpleGradientBoostingRegressor

def train_demand_forecaster(data_path="data/dataset_15min.csv", model_out="models/demand_forecast.json"):
    print("[Forecasting] Training Demand Load Model (t+1h horizon)...")
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    n = len(rows)
    # Features: [hour, day_of_week, is_weekend, ambient_temp, humidity, load_kw_lag1]
    X_list = []
    y_list = []
    
    for i in range(4, n):
        r = rows[i]
        r_lag1 = rows[i - 1]
        r_lag4 = rows[i - 4]
        
        feats = [
            float(r["hour"]),
            float(r["day_of_week"]),
            float(r["is_weekend"]),
            float(r["ambient_temp"]),
            float(r["humidity"]),
            float(r_lag1["load_kw"]),
            float(r_lag4["load_kw"])
        ]
        target = float(r["load_fc_1h"])
        X_list.append(feats)
        y_list.append(target)
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    model = SimpleGradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    print(f"[Demand Model] Evaluation Test MAE: {mae:.3f} kW | RMSE: {rmse:.3f} kW")
    
    model.save(model_out)
    print(f"[Demand Model] Saved successfully to {model_out}")
    return mae, rmse

if __name__ == "__main__":
    train_demand_forecaster()
