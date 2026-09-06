"""
Feature engineering for PM2.5 forecasting.

Builds lag features, rolling statistics, and cyclic time encodings for
a single station (spatio-temporal features per station), and can stack
all stations into one modelling table.
"""
import numpy as np
import pandas as pd

STATIONS = [
    "Pusa", "IBHAS_Dilshad_garden", "Sirifort", "Jahangirpur",
    "Najafgarh", "Wazirpur", "Vivek_vihar", "Mundka",
    "Sri_Aurbuindo_Marg", "Chandani_Chowk", "Huda_Sector_fatehabad",
]

LAGS = [1, 3, 6, 24]
ROLLING_WINDOWS = [3, 24]
HORIZON = 1  # forecast PM2.5, HORIZON hours ahead


def station_frame(wide_df: pd.DataFrame, station: str) -> pd.DataFrame:
    cols = [c for c in wide_df.columns if c.startswith(f"{station}__")]
    sub = wide_df[cols].copy()
    sub.columns = [c.replace(f"{station}__", "") for c in sub.columns]
    return sub


def build_features(wide_df: pd.DataFrame, station: str) -> pd.DataFrame:
    df = station_frame(wide_df, station)
    if "PM2.5" not in df.columns:
        raise ValueError(f"No PM2.5 column for station {station}")

    feat = pd.DataFrame(index=df.index)
    feat["pm25"] = df["PM2.5"]

    # lag features
    for lag in LAGS:
        feat[f"pm25_lag{lag}"] = df["PM2.5"].shift(lag)

    # rolling stats (shifted by 1 so they only use past info)
    for w in ROLLING_WINDOWS:
        feat[f"pm25_roll_mean_{w}"] = df["PM2.5"].shift(1).rolling(w).mean()
        feat[f"pm25_roll_std_{w}"] = df["PM2.5"].shift(1).rolling(w).std()

    # weather features (same-hour, since weather is a driver not a leak)
    for col in ["AT", "RH", "WS", "WD", "SR", "BP"]:
        if col in df.columns:
            feat[col] = df[col]

    # other pollutants as cross-features (lag 1 to avoid leakage with target)
    for col in ["NO2", "CO", "Ozone", "PM10"]:
        if col in df.columns:
            feat[f"{col}_lag1"] = df[col].shift(1)

    # cyclic time encodings
    feat["hour_sin"] = np.sin(2 * np.pi * feat.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat.index.hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * feat.index.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat.index.month / 12)
    feat["dow"] = feat.index.dayofweek

    feat["station"] = station

    # target: PM2.5 HORIZON hours ahead
    feat["target"] = df["PM2.5"].shift(-HORIZON)

    return feat


def build_all_stations(wide_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for st in STATIONS:
        try:
            frames.append(build_features(wide_df, st))
        except ValueError:
            continue
    all_feat = pd.concat(frames)
    all_feat = pd.get_dummies(all_feat, columns=["station"], prefix="st")
    all_feat = all_feat.dropna()
    return all_feat
