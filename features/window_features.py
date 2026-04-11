from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


CSV_COLUMNS = ["timestamp", "temp_C", "humidity", "eco2_ppm", "tvoc"]


def load_realtime_data(csv_path: str | "pd.PathLike[str]" | None) -> pd.DataFrame:
    """
    Load realtime sensor CSV.

    If the file doesn't exist yet, returns an empty DataFrame with expected columns.
    """
    if csv_path is None:
        return pd.DataFrame(columns=CSV_COLUMNS)

    p = str(csv_path)
    try:
        df = pd.read_csv(p)
    except FileNotFoundError:
        return pd.DataFrame(columns=CSV_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=CSV_COLUMNS)

    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=CSV_COLUMNS)

    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    df = df[CSV_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


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
        for var in ["temp_C", "humidity", "eco2_ppm", "tvoc"]:
            for stat in ["mean", "std", "min", "max", "last", "slope", "delta_last_first"]:
                base[f"{var}_{stat}"] = 0.0
        base["tvoc_spike_flag"] = 0.0
        base["eco2_high_fraction"] = 0.0
        base["humidity_out_of_range_fraction"] = 0.0
        return base

    df_window = df_window.sort_values("timestamp")

    out: Dict[str, float] = {}
    for var in ["temp_C", "humidity", "eco2_ppm", "tvoc"]:
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

