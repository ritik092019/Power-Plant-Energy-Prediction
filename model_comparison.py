"""Compare Linear Regression, Random Forest, and the trained PyTorch ANN."""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "powerplant_data.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
SEED = 42

class ANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.network(x)

def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Place powerplant_data.csv beside this script.")
    config_path = MODEL_DIR / "model_config.json"
    weights_path = MODEL_DIR / "best_para.pt"
    scaler_path = MODEL_DIR / "scaler.pkl"
    if not all(p.exists() for p in (config_path, weights_path, scaler_path)):
        raise FileNotFoundError("Run python ann_regression.py first to train and save the ANN.")

    df = pd.read_csv(DATA_PATH).dropna()
    X = df.drop(columns=["PE"])
    y = df["PE"].astype(float)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED
    )

    # Baselines use the same training and test rows as the ANN.
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    rf = RandomForestRegressor(
        n_estimators=200, random_state=SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    ann = ANN(input_size=X.shape[1])
    ann.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    ann.eval()
    with torch.no_grad():
        ann_pred = ann(torch.from_numpy(X_test_scaled)).numpy().ravel()

    predictions = {
        "Linear Regression": linear.predict(X_test),
        "Random Forest": rf.predict(X_test),
        "ANN (PyTorch)": ann_pred
    }

    rows = []
    for name, pred in predictions.items():
        row = {"Model": name, **metrics(y_test.to_numpy(), pred)}
        rows.append(row)
    results = pd.DataFrame(rows).sort_values("RMSE")
    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison (lower error is better; higher R2 is better):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    ax = results.set_index("Model")[["RMSE", "MAE"]].plot(
        kind="bar", figsize=(9, 5), rot=15
    )
    ax.set_ylabel("Error")
    ax.set_title("Model Error Comparison")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
