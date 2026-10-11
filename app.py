"""Streamlit app for interactive power plant output prediction."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
CONFIG_PATH = MODEL_DIR / "model_config.json"
WEIGHTS_PATH = MODEL_DIR / "best_para.pt"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METRICS_PATH = OUTPUT_DIR / "ann_metrics.json"

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

st.set_page_config(page_title="Power Plant Output Prediction", page_icon="⚡", layout="centered")
st.title("⚡ Power Plant Energy Output Prediction")
st.write(
    "Predict electrical power output (PE) from the plant's input measurements "
    "using a trained PyTorch Artificial Neural Network."
)

required = [CONFIG_PATH, WEIGHTS_PATH, SCALER_PATH]
if not all(path.exists() for path in required):
    st.error("Trained model files are missing. Run `python ann_regression.py` first.")
    st.stop()

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)
features = config["input_features"]
scaler = joblib.load(SCALER_PATH)

@st.cache_resource
def load_model(input_size, weights_file):
    model = ANN(input_size)
    model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
    model.eval()
    return model

model = load_model(len(features), str(WEIGHTS_PATH))

st.subheader("Enter input measurements")
st.caption("The input feature names are taken directly from your CSV dataset.")
defaults = {
    "AT": (15.0, -10.0, 40.0),
    "V": (40.0, 25.0, 65.0),
    "AP": (1013.0, 950.0, 1050.0),
    "RH": (60.0, 20.0, 100.0)
}
values = {}
columns = st.columns(2)
for index, feature in enumerate(features):
    default, minimum, maximum = defaults.get(feature, (0.0, -1_000_000.0, 1_000_000.0))
    with columns[index % 2]:
        values[feature] = st.number_input(
            feature,
            value=float(default),
            min_value=float(minimum),
            max_value=float(maximum),
            help=f"Input feature: {feature}"
        )

if st.button("Predict Power Output", type="primary", use_container_width=True):
    input_df = pd.DataFrame([[values[f] for f in features]], columns=features)
    scaled = scaler.transform(input_df).astype(np.float32)
    with torch.no_grad():
        prediction = model(torch.from_numpy(scaled)).item()
    st.success(f"Predicted PE: {prediction:.2f}")
    st.caption("Prediction is in the target column's units. Confirm units and valid input ranges for your dataset.")

if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    st.subheader("ANN test-set performance")
    cols = st.columns(4)
    for col, key in zip(cols, ["MAE", "RMSE", "MSE", "R2"]):
        col.metric(key, f"{metrics[key]:.4f}")

st.subheader("Training results")
for filename, title in [
    ("loss_curve.png", "Training and validation loss"),
    ("actual_vs_predicted.png", "Actual vs predicted values"),
    ("residual_plot.png", "Residual plot")
]:
    path = OUTPUT_DIR / filename
    if path.exists():
        st.image(str(path), caption=title, use_container_width=True)

st.caption("Educational mini project. Predictions are estimates and should not be treated as operational control advice.")
