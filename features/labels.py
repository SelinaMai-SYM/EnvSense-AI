from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOM_RESET_ACTIONS = ["Stay", "Open window", "Open door", "Move soon"]
SLEEP_READINESS = ["Good to sleep", "Sleep okay after ventilating", "Not ideal yet"]
SLEEP_FEEDBACK_VALUES = {"slept_well", "okay", "poor_sleep"}


def simulate_environment_series(
    *,
    duration_minutes: int,
    sample_interval_sec: int,
    seed: int = 42,
    scenario: str = "study",
) -> pd.DataFrame:
    """
    Generate a realistic simulated time series for end-to-end operation.

    scenario:
      - "study": more pronounced occupancy bursts
      - "sleep": slightly lower occupancy but more "sleep readiness" degradations
    """
    rng = np.random.RandomState(seed)

    steps = int(duration_minutes * 60 // sample_interval_sec)
    if steps < 2:
        steps = 2

    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=duration_minutes)

    timestamps = [start + timedelta(seconds=i * sample_interval_sec) for i in range(steps)]

    # Temperature baseline and noise
    base_temp = 22.0 if scenario == "study" else 21.5
    temp_amp = 1.1 if scenario == "study" else 0.8
    temp = base_temp + temp_amp * np.sin(np.linspace(0, 2 * math.pi, steps)) + rng.normal(0.0, 0.12, size=steps)

    # Humidity is coupled to temperature with lag/noise
    humidity_base = 47.0 if scenario == "study" else 50.0
    humidity = humidity_base + 6.0 * np.sin(np.linspace(0, 2 * math.pi, steps) + 0.4) - 1.0 * (temp - base_temp)
    humidity += rng.normal(0.0, 1.2, size=steps)

    # Occupancy / metabolic activity signal
    if scenario == "study":
        occ = 0.5 * (1 + np.sin(np.linspace(0, 4 * math.pi, steps) - 0.7))  # stronger variation
    else:
        occ = 0.45 * (1 + np.sin(np.linspace(0, 2.0 * math.pi, steps) - 1.1))

    # eCO2 dynamics (relax towards an occupancy-dependent target)
    eco2 = np.zeros(steps, dtype=float)
    eco2[0] = 750.0 + 60.0 * occ[0]

    # For "sleep" scenario, allow longer high eco2 stretches
    eco2_baseline = 640.0 if scenario == "study" else 680.0
    eco2_span = 460.0 if scenario == "study" else 540.0

    for i in range(1, steps):
        occ_strength = float(occ[i])
        target = eco2_baseline + eco2_span * occ_strength

        # Occasional ventilation: when occ is low, eco2 drops faster
        relax = 0.06 if occ_strength > 0.5 else 0.1
        eco2[i] = eco2[i - 1] + relax * (target - eco2[i - 1]) + rng.normal(0.0, 9.0)

        # Spikes during high occupancy
        if occ_strength > 0.72 and rng.rand() < 0.03:
            eco2[i] += rng.uniform(80.0, 180.0)

    # TVOC dynamics correlated with occ and eco2 excursions
    tvoc = np.zeros(steps, dtype=float)
    tvoc[0] = 130.0 + 30.0 * occ[0]

    tvoc_baseline = 110.0 if scenario == "study" else 125.0
    for i in range(1, steps):
        occ_strength = float(occ[i])
        tvoc_target = tvoc_baseline + 200.0 * occ_strength + max(0.0, eco2[i] - 900.0) * 0.08
        relax = 0.08
        tvoc[i] = tvoc[i - 1] + relax * (tvoc_target - tvoc[i - 1]) + rng.normal(0.0, 4.5)
        if rng.rand() < 0.01:
            tvoc[i] += rng.uniform(80.0, 200.0)

    # Clamp
    temp = np.clip(temp, 15.0, 35.0)
    humidity = np.clip(humidity, 20.0, 90.0)
    eco2 = np.clip(eco2, 400.0, 2200.0)
    tvoc = np.clip(tvoc, 20.0, 1200.0)

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "temp_C": temp.astype(float),
            "humidity": humidity.astype(float),
            "eco2_ppm": eco2.astype(float),
            "tvoc": tvoc.astype(float),
        }
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def derive_room_reset_best_action(df_past: pd.DataFrame, df_future: pd.DataFrame) -> str:
    """
    Heuristic label derivation for Room Reset Coach bootstrap data.
    """
    if df_past is None or df_past.empty:
        return "Move soon"

    eco2_last = float(df_past["eco2_ppm"].iloc[-1])
    eco2_future_mean = float(df_future["eco2_ppm"].mean()) if df_future is not None and not df_future.empty else eco2_last
    eco2_delta = eco2_future_mean - eco2_last

    tvoc = pd.to_numeric(df_past["tvoc"], errors="coerce").dropna().to_numpy(dtype=float)
    tvoc_mean = float(np.mean(tvoc)) if tvoc.size else 0.0
    tvoc_std = float(np.std(tvoc, ddof=0)) if tvoc.size else 0.0
    tvoc_last = float(tvoc[-1]) if tvoc.size else 0.0
    tvoc_spike = bool(tvoc_last > tvoc_mean + 1.2 * tvoc_std and (tvoc.max() - tvoc.min()) > 80) if tvoc.size else False

    temp_last = float(df_past["temp_C"].iloc[-1])
    humidity_last = float(df_past["humidity"].iloc[-1])

    temp_ok = 20.0 <= temp_last <= 26.0
    humidity_ok = 35.0 <= humidity_last <= 55.0

    # Label rules (matching the user requirements)
    if eco2_last >= 1050.0 and eco2_delta > 90.0:
        return "Open window"

    # Moderate eco2 + tvoc spike => open door
    if (850.0 <= eco2_last < 1200.0) and tvoc_spike:
        return "Open door"

    # All stable and acceptable => stay
    if (eco2_last < 950.0) and (abs(eco2_delta) < 45.0) and temp_ok and humidity_ok and not tvoc_spike:
        return "Stay"

    # Otherwise infer likely need to leave/ventilate quickly
    return "Move soon"


def derive_sleep_readiness(df_window: pd.DataFrame) -> str:
    """
    Heuristic label derivation for Sleep Guard.
    """
    if df_window is None or df_window.empty:
        return "Not ideal yet"

    eco2 = pd.to_numeric(df_window["eco2_ppm"], errors="coerce").dropna().to_numpy(dtype=float)
    temp = pd.to_numeric(df_window["temp_C"], errors="coerce").dropna().to_numpy(dtype=float)
    humidity = pd.to_numeric(df_window["humidity"], errors="coerce").dropna().to_numpy(dtype=float)

    eco2_mean = float(np.mean(eco2)) if eco2.size else 1500.0
    eco2_max = float(np.max(eco2)) if eco2.size else 1600.0
    eco2_slope = float(np.polyfit(np.arange(eco2.size, dtype=float), eco2, 1)[0]) if eco2.size >= 2 else 0.0

    temp_mean = float(np.mean(temp)) if temp.size else 23.0
    humidity_mean = float(np.mean(humidity)) if humidity.size else 45.0

    temp_std = float(np.std(temp, ddof=0)) if temp.size else 2.0
    humidity_std = float(np.std(humidity, ddof=0)) if humidity.size else 8.0

    thermal_ok = (19.0 <= temp_mean <= 25.5) and (35.0 <= humidity_mean <= 55.0)
    thermal_stable = (temp_std <= 1.6) and (humidity_std <= 7.0)

    # Core eco2-driven sleep readiness
    if eco2_mean <= 900.0 and eco2_max <= 1100.0 and eco2_slope <= 8.0 and thermal_ok and thermal_stable:
        return "Good to sleep"

    if eco2_mean <= 1050.0 and eco2_max <= 1300.0 and thermal_ok and (thermal_stable or abs(eco2_slope) <= 12.0):
        # If eco2 is borderline but not terrible, user can ventilate first.
        return "Sleep okay after ventilating"

    # Degraded conditions: prolonged high eco2 and/or thermal instability
    return "Not ideal yet"


def map_sleep_feedback(raw_value: str) -> Optional[str]:
    """
    Map optional sleep feedback labels to sleep readiness classes.
    Input labels:
      ["slept_well", "okay", "poor_sleep"]
    Output classes:
      ["Good to sleep", "Sleep okay after ventilating", "Not ideal yet"]
    """
    if not raw_value:
        return None
    k = str(raw_value).strip().lower()
    if k == "slept_well":
        return "Good to sleep"
    if k == "okay":
        return "Sleep okay after ventilating"
    if k == "poor_sleep":
        return "Not ideal yet"
    return None


def _iter_candidate_annotation_files(source: str | Path) -> List[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if not path.exists() or not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".csv")


def load_room_reset_session_action_labels(source: str | Path) -> pd.DataFrame:
    """
    Load best-action annotations from a CSV file or a directory that contains one.
    """
    candidate_files = _iter_candidate_annotation_files(source)
    if not candidate_files:
        return pd.DataFrame(columns=["timestamp", "best_action"])

    for candidate in candidate_files:
        try:
            df = pd.read_csv(candidate)
        except Exception:
            continue

        if "timestamp" not in df.columns:
            continue

        label_column = None
        for alt in ["best_action", "action", "label"]:
            if alt in df.columns:
                label_column = alt
                break
        if label_column is None:
            continue

        if label_column != "best_action":
            df = df.rename(columns={label_column: "best_action"})

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df[["timestamp", "best_action"]].copy()
        return df

    return pd.DataFrame(columns=["timestamp", "best_action"])


def load_sleep_feedback_labels(source: str | Path) -> pd.DataFrame:
    """
    Load sleep-outcome feedback from a CSV file or a directory that contains one.
    """
    candidate_files = _iter_candidate_annotation_files(source)
    if not candidate_files:
        return pd.DataFrame(columns=["timestamp", "feedback_label", "sleep_readiness"])

    for candidate in candidate_files:
        try:
            df = pd.read_csv(candidate)
        except Exception:
            continue

        if "timestamp" not in df.columns:
            continue

        feedback_column = None
        for column in df.columns:
            if column == "timestamp":
                continue
            sample = df[column].dropna().astype(str).str.strip().str.lower().tolist()
            if sample and set(sample).issubset(SLEEP_FEEDBACK_VALUES):
                feedback_column = str(column)
                break
        if feedback_column is None:
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.rename(columns={feedback_column: "feedback_label"})
        df["sleep_readiness"] = df["feedback_label"].apply(map_sleep_feedback)
        df = df.dropna(subset=["timestamp", "sleep_readiness"])
        return df[["timestamp", "feedback_label", "sleep_readiness"]].copy()

    return pd.DataFrame(columns=["timestamp", "feedback_label", "sleep_readiness"])

