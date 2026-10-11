"""Train a PyTorch ANN to predict power plant electrical output (PE)."""
from pathlib import Path
import json
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "powerplant_data.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 20

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class ANN(nn.Module):
    """Feedforward neural network for tabular regression."""
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

def regression_metrics(actual, predicted):
    return {
        "MSE": float(mean_squared_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted))
    }

def main():
    set_seed()
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place powerplant_data.csv beside ann_regression.py."
        )

    df = pd.read_csv(DATA_PATH)
    if "PE" not in df.columns:
        raise ValueError("The dataset must contain a target column named 'PE'.")

    df = df.dropna().copy()
    X = df.drop(columns=["PE"])
    y = df["PE"].astype(float)
    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        raise ValueError("All input feature columns must be numeric.")

    # Split: 70% training, 15% validation, 15% final testing.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED
    )

    # Fit scaler only on training data to prevent data leakage.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_val_np = y_val.to_numpy(dtype=np.float32).reshape(-1, 1)

    train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train_np))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    X_val_t = torch.from_numpy(X_val_s)
    y_val_t = torch.from_numpy(y_val_np)
    X_test_t = torch.from_numpy(X_test_s)

    model = ANN(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    train_history, val_history = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_path = MODEL_DIR / "best_para.pt"

    for epoch in range(EPOCHS):
        model.train()
        train_total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            predictions = model(xb)
            loss = criterion(predictions, yb)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(xb)

        epoch_train_loss = train_total / len(train_ds)
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            epoch_val_loss = criterion(val_predictions, y_val_t).item()

        train_history.append(epoch_train_loss)
        val_history.append(epoch_val_loss)
        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
            f"Train MSE: {epoch_train_loss:.4f} | Val MSE: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    # Load the best validation checkpoint before final testing.
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_t).numpy().ravel()

    metrics = regression_metrics(y_test.to_numpy(), test_predictions)
    metrics["best_epoch"] = best_epoch
    metrics["best_validation_mse"] = best_val_loss
    metrics["train_rows"] = int(len(X_train))
    metrics["validation_rows"] = int(len(X_val))
    metrics["test_rows"] = int(len(X_test))

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({"input_features": list(X.columns), "target": "PE"}, f, indent=2)
    with open(OUTPUT_DIR / "ann_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "Actual Values": y_test.to_numpy(),
        "Predicted Values": test_predictions,
        "Residual": y_test.to_numpy() - test_predictions
    }).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    # Plot training and validation loss.
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(train_history) + 1), train_history, label="Training MSE")
    plt.plot(range(1, len(val_history) + 1), val_history, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("ANN Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    plt.close()

    # Actual-versus-predicted plot.
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, test_predictions, alpha=0.5)
    lower = min(float(y_test.min()), float(test_predictions.min()))
    upper = max(float(y_test.max()), float(test_predictions.max()))
    plt.plot([lower, upper], [lower, upper], "r--", label="Ideal prediction")
    plt.xlabel("Actual PE")
    plt.ylabel("Predicted PE")
    plt.title("Actual vs Predicted Power Output")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Residual plot.
    residuals = y_test.to_numpy() - test_predictions
    plt.figure(figsize=(8, 5))
    plt.scatter(test_predictions, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted PE")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("ANN Residual Plot")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=150)
    plt.close()

    print("\nFinal ANN test metrics:")
    for key in ("MSE", "RMSE", "MAE", "R2"):
        print(f"{key}: {metrics[key]:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"\nSaved model and scaler in: {MODEL_DIR}")
    print(f"Saved results and graphs in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
