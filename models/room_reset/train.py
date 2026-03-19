from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from features.labels import ROOM_RESET_ACTIONS, derive_room_reset_best_action, simulate_environment_series
from features.window_features import compute_window_features, load_realtime_data
from utils.io_utils import load_hardware_config


def _extract_training_from_realtime(
    df: pd.DataFrame,
    *,
    sample_interval_sec: int,
    window_minutes: int = 5,
    horizon_minutes: int = 2,
    stride: int = 2,
) -> Tuple[List[Dict[str, float]], List[str]]:
    window_steps = max(2, int(window_minutes * 60 // sample_interval_sec))
    horizon_steps = max(1, int(horizon_minutes * 60 // sample_interval_sec))

    X_rows: List[Dict[str, float]] = []
    y: List[str] = []

    if df.empty or len(df) < window_steps + horizon_steps + 5:
        return X_rows, y

    df = df.sort_values("timestamp").reset_index(drop=True)
    for idx in range(window_steps, len(df) - horizon_steps, stride):
        df_past = df.iloc[idx - window_steps : idx]
        df_future = df.iloc[idx : idx + horizon_steps]
        label = derive_room_reset_best_action(df_past, df_future)
        feats = compute_window_features(df_past)
        X_rows.append(feats)
        y.append(label)

    return X_rows, y


def _bootstrap_synthetic_training(
    *,
    sample_interval_sec: int,
    n_sequences: int = 8,
    duration_minutes: int = 200,
    window_minutes: int = 5,
    horizon_minutes: int = 2,
) -> Tuple[List[Dict[str, float]], List[str]]:
    X_rows: List[Dict[str, float]] = []
    y: List[str] = []

    window_steps = max(2, int(window_minutes * 60 // sample_interval_sec))
    horizon_steps = max(1, int(horizon_minutes * 60 // sample_interval_sec))
    stride = max(1, window_steps // 6)

    for i in range(n_sequences):
        seed = 1000 + i * 17
        df = simulate_environment_series(
            duration_minutes=duration_minutes,
            sample_interval_sec=sample_interval_sec,
            seed=seed,
            scenario="study",
        ).sort_values("timestamp").reset_index(drop=True)

        if len(df) < window_steps + horizon_steps + 5:
            continue

        for idx in range(window_steps, len(df) - horizon_steps, stride):
            df_past = df.iloc[idx - window_steps : idx]
            df_future = df.iloc[idx : idx + horizon_steps]
            label = derive_room_reset_best_action(df_past, df_future)
            feats = compute_window_features(df_past)
            X_rows.append(feats)
            y.append(label)

    return X_rows, y


def train_room_reset_model(
    *,
    realtime_csv_path: str | Path,
    model_path: str | Path,
    force_synthetic: bool = False,
) -> Path:
    """
    Train a lightweight RandomForest model for Room Reset Coach.

    If realtime training data is missing/insufficient, synthetic bootstrap data is used.
    """
    realtime_csv_path = Path(realtime_csv_path)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_hardware_config()
    sample_interval_sec = int(cfg.get("sample_interval_sec", 10))

    df = load_realtime_data(str(realtime_csv_path))

    X_rows: List[Dict[str, float]]
    y: List[str]
    X_rows, y = ([], [])

    if not force_synthetic:
        X_rows, y = _extract_training_from_realtime(df, sample_interval_sec=sample_interval_sec)

    # Ensure we always have enough samples for training
    min_samples = 250
    if len(y) < min_samples:
        X_syn, y_syn = _bootstrap_synthetic_training(sample_interval_sec=sample_interval_sec)
        X_rows = X_rows + X_syn
        y = y + y_syn

    if len(y) == 0:
        # Absolute fallback: train on tiny synthetic data
        X_syn, y_syn = _bootstrap_synthetic_training(sample_interval_sec=sample_interval_sec, n_sequences=3, duration_minutes=120)
        X_rows = X_syn
        y = y_syn

    feature_names = sorted(X_rows[0].keys()) if X_rows else []

    X = pd.DataFrame([{k: row.get(k, 0.0) for k in feature_names} for row in X_rows], columns=feature_names)
    y_arr = np.array(y, dtype=object)

    clf = RandomForestClassifier(
        n_estimators=240,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=2,
    )
    clf.fit(X, y_arr)

    artifact = {"model": clf, "feature_names": feature_names, "label_names": ROOM_RESET_ACTIONS}
    joblib.dump(artifact, model_path)
    return model_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    train_room_reset_model(
        realtime_csv_path=root / "data" / "realtime.csv",
        model_path=root / "models" / "room_reset" / "model.joblib",
    )

