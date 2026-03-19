from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from features.labels import SLEEP_READINESS, derive_sleep_readiness, simulate_environment_series
from features.window_features import compute_window_features, load_realtime_data
from utils.io_utils import load_hardware_config


def _extract_training_from_realtime(
    df: pd.DataFrame,
    *,
    sample_interval_sec: int,
    window_minutes: int = 30,
    stride: int = 3,
) -> Tuple[List[Dict[str, float]], List[str]]:
    window_steps = max(2, int(window_minutes * 60 // sample_interval_sec))
    X_rows: List[Dict[str, float]] = []
    y: List[str] = []

    if df.empty or len(df) < window_steps + 5:
        return X_rows, y

    df = df.sort_values("timestamp").reset_index(drop=True)
    for idx in range(window_steps, len(df), stride):
        df_window = df.iloc[idx - window_steps : idx]
        label = derive_sleep_readiness(df_window)
        feats = compute_window_features(df_window)
        X_rows.append(feats)
        y.append(label)
    return X_rows, y


def _bootstrap_synthetic_training(
    *,
    sample_interval_sec: int,
    n_sequences: int = 6,
    duration_minutes: int = 260,
    window_minutes: int = 30,
) -> Tuple[List[Dict[str, float]], List[str]]:
    window_steps = max(2, int(window_minutes * 60 // sample_interval_sec))
    stride = max(1, window_steps // 6)

    X_rows: List[Dict[str, float]] = []
    y: List[str] = []

    for i in range(n_sequences):
        seed = 2000 + i * 29
        df = simulate_environment_series(
            duration_minutes=duration_minutes,
            sample_interval_sec=sample_interval_sec,
            seed=seed,
            scenario="sleep",
        ).sort_values("timestamp").reset_index(drop=True)

        if len(df) < window_steps + 5:
            continue

        for idx in range(window_steps, len(df), stride):
            df_window = df.iloc[idx - window_steps : idx]
            label = derive_sleep_readiness(df_window)
            feats = compute_window_features(df_window)
            X_rows.append(feats)
            y.append(label)

    return X_rows, y


def train_sleep_guard_model(
    *,
    realtime_csv_path: str | Path,
    model_path: str | Path,
    force_synthetic: bool = False,
) -> Path:
    """
    Train a lightweight RandomForest model for Dorm Sleep Guard.
    """
    realtime_csv_path = Path(realtime_csv_path)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_hardware_config()
    sample_interval_sec = int(cfg.get("sample_interval_sec", 10))

    df = load_realtime_data(str(realtime_csv_path))

    if not force_synthetic:
        X_rows, y = _extract_training_from_realtime(df, sample_interval_sec=sample_interval_sec)
    else:
        X_rows, y = [], []

    min_samples = 220
    if len(y) < min_samples:
        X_syn, y_syn = _bootstrap_synthetic_training(sample_interval_sec=sample_interval_sec)
        X_rows = X_rows + X_syn
        y = y + y_syn

    if len(y) == 0:
        X_syn, y_syn = _bootstrap_synthetic_training(sample_interval_sec=sample_interval_sec, n_sequences=3, duration_minutes=200)
        X_rows, y = X_syn, y_syn

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

    artifact = {"model": clf, "feature_names": feature_names, "label_names": SLEEP_READINESS}
    joblib.dump(artifact, model_path)
    return model_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    train_sleep_guard_model(
        realtime_csv_path=root / "data" / "realtime.csv",
        model_path=root / "models" / "sleep_guard" / "model.joblib",
    )

