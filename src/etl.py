"""
ETL pipeline for Delhi CPCB Air Quality dataset.

Reads the raw multi-station wide-format Excel export, cleans it, and
reshapes it into a tidy long-format table:

    timestamp | station | metric | value

Also produces a wide per-station-pollutant table used later for
feature engineering and modelling (one row per timestamp, one column
per station_pollutant combination), which is more convenient for
forecasting.

Usage:
    python src/etl.py --input <path_to_raw.xlsx> --outdir data/
"""
import argparse
import re
import numpy as np
import pandas as pd
import openpyxl

STATIONS = [
    "Pusa", "IBHAS_Dilshad_garden", "Sirifort", "Jahangirpur",
    "Najafgarh", "Wazirpur", "Vivek_vihar", "Mundka",
    "Sri_Aurbuindo_Marg", "Chandani_Chowk", "Huda_Sector_fatehabad",
]
METRICS_PER_STATION = 24

METRIC_UNIT_RE = re.compile(r"\s*\(([^)]*)\)\s*$")


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw workbook fast, using read-only iteration
    (pandas.read_excel is far slower on a 96MB / 265-column sheet)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    data = list(rows)
    df = pd.DataFrame(data, columns=header)
    return df


def build_metric_names(header) -> list[str]:
    """Re-derive a clean (station, metric) label for every data column,
    ignoring the raw header text (which has inconsistent / duplicated
    station prefixes in a few places) and instead relying on the fixed
    24-metrics-per-station block structure confirmed during EDA."""
    # Metric names taken from the first station's block, which is clean.
    first_block = header[1:1 + METRICS_PER_STATION]
    metric_names = []
    for col in first_block:
        name = col.split("Pusa_", 1)[-1]
        name = METRIC_UNIT_RE.sub("", name).strip()
        metric_names.append(name)
    return metric_names


def reshape_long(df: pd.DataFrame) -> pd.DataFrame:
    header = list(df.columns)
    metric_names = build_metric_names(header)

    frames = []
    col_idx = 1  # column 0 is Timestamp
    for station in STATIONS:
        block = df.iloc[:, col_idx: col_idx + METRICS_PER_STATION].copy()
        block.columns = metric_names
        block.insert(0, "station", station)
        block.insert(0, "timestamp", df["Timestamp"])
        frames.append(block)
        col_idx += METRICS_PER_STATION

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.melt(
        id_vars=["timestamp", "station"], var_name="metric", value_name="value"
    )
    long_df["value"] = pd.to_numeric(
        long_df["value"].replace("NA", np.nan), errors="coerce"
    )
    return long_df


def reshape_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp, columns = station__metric. Built directly
    from the raw wide dataframe (much cheaper than melt+pivot on 18M rows)."""
    header = list(df.columns)
    metric_names = build_metric_names(header)

    out = {"timestamp": df["Timestamp"]}
    col_idx = 1
    for station in STATIONS:
        block = df.iloc[:, col_idx: col_idx + METRICS_PER_STATION]
        for metric, series in zip(metric_names, block.columns):
            out[f"{station}__{metric}"] = pd.to_numeric(
                block[series].replace("NA", np.nan), errors="coerce"
            )
        col_idx += METRICS_PER_STATION

    wide = pd.DataFrame(out).set_index("timestamp").sort_index()
    return wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    print("Loading raw workbook...")
    raw = load_raw(args.input)
    print(f"Raw shape: {raw.shape}")

    print("Reshaping to wide format...")
    wide_df = reshape_wide(raw)
    wide_path = f"{args.outdir}/aq_wide.parquet"
    wide_df.to_parquet(wide_path)
    print(f"Saved {wide_path} ({wide_df.shape})")

    # Small CSV sample for the repo (full data is too large for GitHub)
    sample_path = f"{args.outdir}/aq_sample.csv"
    wide_df.tail(24 * 30).to_csv(sample_path)  # last 30 days
    print(f"Saved {sample_path} (last 30 days, for repo/demo use)")


if __name__ == "__main__":
    main()
