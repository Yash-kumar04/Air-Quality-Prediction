"""
Streamlit prototype: Delhi Air Quality forecasting & hotspot dashboard.

Loads the trained XGBoost model and recent station data, forecasts
next-hour PM2.5 for every station, and flags hotspots (stations whose
forecast crosses CPCB AQI breakpoints).

Run with:
    streamlit run app/dashboard.py
"""
import json
import sys
import os

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features import build_all_stations, STATIONS  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "aq_wide.parquet")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_pm25.json")

# Approximate station coordinates (Delhi NCR) for the hotspot map
COORDS = {
    "Pusa": (28.6389, 77.1631),
    "IBHAS_Dilshad_garden": (28.6789, 77.3222),
    "Sirifort": (28.5504, 77.2160),
    "Jahangirpur": (28.7286, 77.1687),
    "Najafgarh": (28.6096, 76.9791),
    "Wazirpur": (28.6996, 77.1626),
    "Vivek_vihar": (28.6725, 77.3151),
    "Mundka": (28.6819, 77.0298),
    "Sri_Aurbuindo_Marg": (28.5504, 77.1912),
    "Chandani_Chowk": (28.6506, 77.2303),
    "Huda_Sector_fatehabad": (28.4595, 77.0266),
}

AQI_BANDS = [
    (0, 30, "Good", "#2ecc71"),
    (30, 60, "Satisfactory", "#a3d139"),
    (60, 90, "Moderate", "#f5d800"),
    (90, 120, "Poor", "#f39c12"),
    (120, 250, "Very Poor", "#e74c3c"),
    (250, 10_000, "Severe", "#8e44ad"),
]


def aqi_band(pm25: float):
    for lo, hi, label, color in AQI_BANDS:
        if lo <= pm25 < hi:
            return label, color
    return "Unknown", "#7f8c8d"


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def main():
    st.set_page_config(page_title="Delhi AQ Forecast & Hotspots", layout="wide")
    st.title("🌫️ Delhi Air Quality — Forecast & Hotspot Prototype")
    st.caption(
        "XGBoost-based next-hour PM2.5 forecast across 11 CPCB stations, "
        "trained on 2017–2024 hourly sensor + meteorological data."
    )

    wide = load_data()
    model = load_model()

    with st.spinner("Building features and generating forecasts..."):
        feat = build_all_stations(wide)
        latest_ts = feat.index.max()
        latest = feat[feat.index == latest_ts]

        results = []
        for st_name in STATIONS:
            col = f"st_{st_name}"
            if col not in latest.columns:
                continue
            row = latest[latest[col] == 1]
            if row.empty:
                continue
            X_row = row.drop(columns=["target"])
            pred = model.predict(X_row)[0]
            actual = row["pm25"].values[0]
            label, color = aqi_band(pred)
            lat, lon = COORDS.get(st_name, (28.61, 77.20))
            results.append({
                "station": st_name.replace("_", " "),
                "latest_pm25": round(float(actual), 1),
                "forecast_next_hour": round(float(pred), 1),
                "aqi_band": label,
                "color": color,
                "lat": lat, "lon": lon,
            })

    res_df = pd.DataFrame(results).sort_values("forecast_next_hour", ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Hotspot map — forecast for {latest_ts}")
        fig = px.scatter_mapbox(
            res_df, lat="lat", lon="lon", size="forecast_next_hour",
            color="forecast_next_hour", color_continuous_scale="Reds",
            hover_name="station",
            hover_data={"latest_pm25": True, "forecast_next_hour": True, "aqi_band": True, "lat": False, "lon": False},
            zoom=9, height=500,
        )
        fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Ranked stations")
        st.dataframe(
            res_df[["station", "latest_pm25", "forecast_next_hour", "aqi_band"]],
            hide_index=True, use_container_width=True,
        )

    st.subheader("⚠️ Flagged hotspots (forecast ≥ 120 µg/m³, 'Very Poor'+)")
    hotspots = res_df[res_df["forecast_next_hour"] >= 120]
    if hotspots.empty:
        st.success("No stations currently forecast in the Very Poor / Severe band.")
    else:
        st.warning(f"{len(hotspots)} station(s) flagged.")
        st.table(hotspots[["station", "forecast_next_hour", "aqi_band"]])

    with open(os.path.join(os.path.dirname(__file__), "..", "outputs", "xgboost_metrics.json")) as f:
        metrics = json.load(f)
    st.caption(
        f"Model performance (held-out test set): MAE={metrics['mae']:.1f} µg/m³, "
        f"RMSE={metrics['rmse']:.1f} µg/m³, R²={metrics['r2']:.3f}"
    )


if __name__ == "__main__":
    main()
