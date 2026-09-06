"""
Train an LSTM to forecast next-hour PM2.5 from a sliding window of past
readings, for a representative station (Pusa — most complete series).

Usage:
    python src/train_lstm.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

from features import station_frame

DATA_PATH = "../data/aq_wide.parquet"
STATION = "Pusa"
WINDOW = 24  # past 24 hours used to predict next hour
MODEL_PATH = "../models/lstm_pm25.keras"
SCALER_PATH = "../models/lstm_scaler.pkl"
METRICS_PATH = "../outputs/lstm_metrics.json"
FIG_DIR = "../outputs/figures"


def make_sequences(values: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    print("Loading data...")
    wide = pd.read_parquet(DATA_PATH)
    wide.index = pd.to_datetime(wide.index)
    df = station_frame(wide, STATION)

    series = df["PM2.5"].interpolate(limit=6).dropna()
    print(f"Series length: {len(series)}")

    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = make_sequences(scaled.flatten(), WINDOW)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split_idx = int(len(X) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(WINDOW, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=256,
        callbacks=[early_stop],
        verbose=2,
    )

    preds_scaled = model.predict(X_test).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    r2 = r2_score(actual, preds)
    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")

    with open(METRICS_PATH, "w") as f:
        json.dump({"mae": mae, "rmse": rmse, "r2": r2,
                   "n_train": len(X_train), "n_test": len(X_test),
                   "station": STATION, "window": WINDOW}, f, indent=2)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # training curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss")
    ax.set_title("LSTM training curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/lstm_training_curve.png")

    # predicted vs actual
    fig, ax = plt.subplots(figsize=(12, 4))
    n = 500
    ax.plot(actual[:n], label="Actual", linewidth=1)
    ax.plot(preds[:n], label="Predicted", linewidth=1, alpha=0.8)
    ax.set_title(f"LSTM ({STATION}): Actual vs Predicted PM2.5 (test set, MAE={mae:.1f})")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/lstm_pred_vs_actual.png")

    print("Done.")


if __name__ == "__main__":
    main()
