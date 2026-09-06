"""
Train an XGBoost regressor to forecast next-hour PM2.5 across all
Delhi stations, using lag/rolling/weather/cyclic features.

Usage:
    python src/train_xgboost.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from features import build_all_stations

DATA_PATH = "../data/aq_wide.parquet"
MODEL_PATH = "../models/xgboost_pm25.json"
METRICS_PATH = "../outputs/xgboost_metrics.json"
FIG_DIR = "../outputs/figures"


def main():
    print("Loading data...")
    wide = pd.read_parquet(DATA_PATH)
    wide.index = pd.to_datetime(wide.index)

    print("Building features for all stations...")
    feat = build_all_stations(wide)
    print(f"Feature table: {feat.shape}")

    X = feat.drop(columns=["target"])
    y = feat["target"]

    # time-based split (last 15% as test, respects temporal order)
    split_idx = int(len(feat) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "n_train": len(X_train), "n_test": len(X_test)}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    model.save_model(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # feature importance plot
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("XGBoost feature importance (top 15)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/xgboost_feature_importance.png")

    # predicted vs actual plot (first 500 test points for readability)
    fig, ax = plt.subplots(figsize=(12, 4))
    n = 500
    ax.plot(y_test.values[:n], label="Actual", linewidth=1)
    ax.plot(preds[:n], label="Predicted", linewidth=1, alpha=0.8)
    ax.set_title(f"XGBoost: Actual vs Predicted PM2.5 (test set, MAE={mae:.1f})")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/xgboost_pred_vs_actual.png")

    print("Done.")


if __name__ == "__main__":
    main()
