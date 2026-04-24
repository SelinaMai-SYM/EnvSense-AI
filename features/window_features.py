from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


CORE_SENSOR_COLUMNS = ["temp_C", "humidity", "eco2_ppm", "tvoc"]
CSV_COLUMNS = ["timestamp", *CORE_SENSOR_COLUMNS]
CALIBRATION_OFFSETS = {
    "temp_C": -1.3,
    "humidity": 0.0,
    "eco2_ppm": 300.0,
    "tvoc": 0.0,
}


def unadjusted_column_name(column: str) -> str:
    return f"{column}_unadjusted"


def adjusted_column_name(column: str) -> str:
    return f"{column}_adjusted"


CALIBRATION_VARIANT_COLUMNS = [
    *(unadjusted_column_name(column) for column in CORE_SENSOR_COLUMNS),
    *(adjusted_column_name(column) for column in CORE_SENSOR_COLUMNS),
]
LOGGER_CSV_COLUMNS = ["timestamp", *CORE_SENSOR_COLUMNS, *CALIBRATION_VARIANT_COLUMNS]


def _empty_sensor_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LOGGER_CSV_COLUMNS)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def apply_sensor_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep paired unadjusted/adjusted traces while exposing calibrated values
    through the canonical sensor columns used by the rest of the pipeline.
    """
    if df is None:
        return _empty_sensor_frame()

    out = df.copy()
    for column in CORE_SENSOR_COLUMNS:
        raw_column = unadjusted_column_name(column)
        adjusted_column = adjusted_column_name(column)
        raw_values = _numeric_series(out, raw_column if raw_column in out.columns else column)
        adjusted_values = raw_values + CALIBRATION_OFFSETS[column]
        out[raw_column] = raw_values
        out[adjusted_column] = adjusted_values
        out[column] = adjusted_values
    return out


def build_calibrated_sensor_row(row: Dict[str, float | str]) -> Dict[str, float | str]:
    """
    Expand one raw sensor row into canonical calibrated fields plus paired
    unadjusted/adjusted variants for traceability.
    """
    out: Dict[str, float | str] = dict(row)
    for column in CORE_SENSOR_COLUMNS:
        raw_value = pd.to_numeric(pd.Series([out.get(column)]), errors="coerce").iloc[0]
        adjusted_value = raw_value + CALIBRATION_OFFSETS[column] if pd.notna(raw_value) else raw_value
        out[unadjusted_column_name(column)] = raw_value
        out[adjusted_column_name(column)] = adjusted_value
        out[column] = adjusted_value
    return out


def load_realtime_data(csv_path: str | "pd.PathLike[str]" | None) -> pd.DataFrame:
    """
    Load realtime sensor CSV.

    If the file doesn't exist yet, returns an empty DataFrame with expected columns.
    """
    if csv_path is None:
        return _empty_sensor_frame()

    p = str(csv_path)
    try:
        df = pd.read_csv(p)
    except FileNotFoundError:
        return _empty_sensor_frame()

    if df.empty:
        return _empty_sensor_frame()

    if "timestamp" not in df.columns:
        return _empty_sensor_frame()

    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = apply_sensor_calibration(df)
    ordered = ["timestamp", *CORE_SENSOR_COLUMNS, *CALIBRATION_VARIANT_COLUMNS]
    remainder = [column for column in df.columns if column not in ordered]
    return df[ordered + remainder]


def get_recent_window(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Return the last `minutes` worth of rows using the latest timestamp in `df`.
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = df.sort_values("timestamp")
    end_ts = df["timestamp"].iloc[-1]
    start_ts = end_ts - pd.Timedelta(minutes=minutes)
    return df[df["timestamp"] >= start_ts].copy()


def _safe_slope(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=float)
    # y = m*x + b
    m = float(np.polyfit(x, y, 1)[0])
    return m


def compute_window_features(df_window: pd.DataFrame) -> Dict[str, float]:
    """
    Compute interpretable summary features for a time window.

    Returns:
      dict of numeric features suitable for RandomForest.
    """
    if df_window is None or df_window.empty:
        # Keep a stable schema for training/inference
        base: Dict[str, float] = {}
        for var in CORE_SENSOR_COLUMNS:
            for stat in ["mean", "std", "min", "max", "last", "slope", "delta_last_first"]:
                base[f"{var}_{stat}"] = 0.0
        base["tvoc_spike_flag"] = 0.0
        base["eco2_high_fraction"] = 0.0
        base["humidity_out_of_range_fraction"] = 0.0
        return base

    df_window = df_window.sort_values("timestamp")

    out: Dict[str, float] = {}
    for var in CORE_SENSOR_COLUMNS:
        arr = pd.to_numeric(df_window.get(var), errors="coerce").dropna().to_numpy(dtype=float)
        if arr.size == 0:
            stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "last": 0.0, "slope": 0.0}
            first = 0.0
            last = 0.0
        else:
            stats = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "last": float(arr[-1]),
                "slope": _safe_slope(arr),
            }
            first = float(arr[0])
            last = float(arr[-1])

        delta = float(last - first)
        out[f"{var}_mean"] = stats["mean"]
        out[f"{var}_std"] = stats["std"]
        out[f"{var}_min"] = stats["min"]
        out[f"{var}_max"] = stats["max"]
        out[f"{var}_last"] = stats["last"]
        out[f"{var}_slope"] = stats["slope"]
        out[f"{var}_delta_last_first"] = delta

    tvoc_arr = pd.to_numeric(df_window.get("tvoc"), errors="coerce").dropna().to_numpy(dtype=float)
    eco2_arr = pd.to_numeric(df_window.get("eco2_ppm"), errors="coerce").dropna().to_numpy(dtype=float)
    humidity_arr = pd.to_numeric(df_window.get("humidity"), errors="coerce").dropna().to_numpy(dtype=float)

    tvoc_spike_flag = 0.0
    if tvoc_arr.size >= 3:
        tvoc_mean = float(np.mean(tvoc_arr))
        tvoc_std = float(np.std(tvoc_arr, ddof=0))
        tvoc_last = float(tvoc_arr[-1])
        tvoc_max = float(np.max(tvoc_arr))

        # Heuristic spike detector: last value significantly above typical level
        if tvoc_last > tvoc_mean + 1.5 * tvoc_std and (tvoc_max - tvoc_mean) > 50:
            tvoc_spike_flag = 1.0
        elif (tvoc_max - tvoc_arr.min()) > 250 and tvoc_last > tvoc_mean + tvoc_std:
            tvoc_spike_flag = 1.0

    # "High eCO2" threshold for indoor environments
    eco2_high_threshold = 1000.0
    eco2_high_fraction = 0.0
    if eco2_arr.size > 0:
        eco2_high_fraction = float(np.mean(eco2_arr >= eco2_high_threshold))

    # Typical comfortable humidity range
    humidity_lo, humidity_hi = 30.0, 60.0
    humidity_out_of_range_fraction = 0.0
    if humidity_arr.size > 0:
        humidity_out_of_range_fraction = float(np.mean((humidity_arr < humidity_lo) | (humidity_arr > humidity_hi)))

    out["tvoc_spike_flag"] = tvoc_spike_flag
    out["eco2_high_fraction"] = eco2_high_fraction
    out["humidity_out_of_range_fraction"] = humidity_out_of_range_fraction

    return out


def compute_window_features_df(df_window: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper to return features as a one-row DataFrame.
    """
    feats = compute_window_features(df_window)
    return pd.DataFrame([feats])

