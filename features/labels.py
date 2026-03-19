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


def simulate_environment_series(
    *,
    duration_minutes: int,
    sample_interval_sec: int,
    seed: int = 42,
    scenario: str = "study",
) -> pd.DataFrame:
    """
    Generate a realistic synthetic time series for end-to-end operation.

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
    Heuristic label derivation for Room Reset Coach (synthetic bootstrapping).
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
    Heuristic label derivation for Dorm Sleep Guard.
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


def map_morning_feedback(morning_feedback: str) -> Optional[str]:
    """
    Map optional morning feedback labels to sleep readiness classes.
    Input labels:
      ["slept_well", "okay", "poor_sleep"]
    Output classes:
      ["Good to sleep", "Sleep okay after ventilating", "Not ideal yet"]
    """
    if not morning_feedback:
        return None
    k = str(morning_feedback).strip().lower()
    if k == "slept_well":
        return "Good to sleep"
    if k == "okay":
        return "Sleep okay after ventilating"
    if k == "poor_sleep":
        return "Not ideal yet"
    return None


def load_room_reset_session_action_labels(session_dir: str | Path) -> pd.DataFrame:
    """
    Optional helper: load best-action labels from session metadata, if present.

    Expected input (one of):
      - `actions.csv` or `labels.csv`
    Required columns:
      - `timestamp` (ISO 8601)
      - `best_action` (one of ROOM_RESET_ACTIONS)
    Returns an empty DataFrame if no usable file is found.
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        return pd.DataFrame(columns=["timestamp", "best_action"])

    candidate_names = ["actions.csv", "labels.csv"]
    label_path: Optional[Path] = None
    for name in candidate_names:
        p = session_dir / name
        if p.exists() and p.is_file():
            label_path = p
            break

    if label_path is None:
        return pd.DataFrame(columns=["timestamp", "best_action"])

    try:
        df = pd.read_csv(label_path)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "best_action"])

    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "best_action"])

    if "best_action" not in df.columns:
        # Allow alternative column naming
        for alt in ["action", "label"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "best_action"})
                break

    if "best_action" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "best_action"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[["timestamp", "best_action"]].copy()
    return df


def load_sleep_morning_feedback(session_dir: str | Path) -> pd.DataFrame:
    """
    Optional helper: load morning feedback labels from session metadata, if present.

    Expected input:
      - `morning_feedback.csv`

    Required columns:
      - `timestamp`
      - `morning_feedback` in {"slept_well","okay","poor_sleep"}
    Returns a DataFrame with an additional mapped column `sleep_readiness`.
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        return pd.DataFrame(columns=["timestamp", "morning_feedback", "sleep_readiness"])

    label_path = session_dir / "morning_feedback.csv"
    if not label_path.exists():
        return pd.DataFrame(columns=["timestamp", "morning_feedback", "sleep_readiness"])

    try:
        df = pd.read_csv(label_path)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "morning_feedback", "sleep_readiness"])

    if "timestamp" not in df.columns or "morning_feedback" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "morning_feedback", "sleep_readiness"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["sleep_readiness"] = df["morning_feedback"].apply(map_morning_feedback)
    df = df.dropna(subset=["timestamp", "sleep_readiness"])
    return df[["timestamp", "morning_feedback", "sleep_readiness"]].copy()

