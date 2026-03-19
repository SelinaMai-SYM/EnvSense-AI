from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from features.labels import (
    ROOM_RESET_ACTIONS,
    derive_room_reset_best_action,
    load_room_reset_session_action_labels,
    simulate_environment_series,
)
from features.window_features import compute_window_features, load_realtime_data
from utils.io_utils import load_hardware_config, repo_root

MODEL_CHOICES = ("random_forest", "gradient_boosting", "hist_gradient_boosting")


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


def _extract_training_from_human_labels(
    df: pd.DataFrame,
    *,
    session_root: str | Path,
    sample_interval_sec: int,
    window_minutes: int = 5,
) -> Tuple[List[Dict[str, float]], List[str]]:
    window_steps = max(2, int(window_minutes * 60 // sample_interval_sec))
    session_root = Path(session_root)
    X_rows: List[Dict[str, float]] = []
    y: List[str] = []

    if df.empty or not session_root.exists():
        return X_rows, y

    df_sorted = df.sort_values("timestamp").reset_index(drop=True).copy()
    df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], utc=True, errors="coerce")
    df_sorted = df_sorted.dropna(subset=["timestamp"])
    if df_sorted.empty:
        return X_rows, y

    for session_dir in sorted(session_root.iterdir()):
        if not session_dir.is_dir():
            continue
        labels_df = load_room_reset_session_action_labels(session_dir)
        if labels_df.empty:
            continue

        labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"], utc=True, errors="coerce")
        labels_df = labels_df.dropna(subset=["timestamp"])
        labels_df = labels_df[labels_df["best_action"].isin(ROOM_RESET_ACTIONS)]
        if labels_df.empty:
            continue

        for _, row in labels_df.iterrows():
            ts = row["timestamp"]
            idx = int(df_sorted["timestamp"].searchsorted(ts, side="right"))
            if idx < window_steps:
                continue
            df_past = df_sorted.iloc[idx - window_steps : idx]
            if len(df_past) < window_steps:
                continue
            X_rows.append(compute_window_features(df_past))
            y.append(str(row["best_action"]))

    return X_rows, y


def train_room_reset_model(
    *,
    realtime_csv_path: str | Path,
    model_path: str | Path,
    force_synthetic: bool = False,
    model_name: str = "random_forest",
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
    session_root = repo_root() / "data" / "room_reset_sessions"

    X_rows: List[Dict[str, float]]
    y: List[str]
    X_rows, y = ([], [])

    X_human, y_human = _extract_training_from_human_labels(
        df,
        session_root=session_root,
        sample_interval_sec=sample_interval_sec,
    )
    if y_human:
        X_rows.extend(X_human)
        y.extend(y_human)

    if not force_synthetic:
        X_auto, y_auto = _extract_training_from_realtime(df, sample_interval_sec=sample_interval_sec)
        X_rows.extend(X_auto)
        y.extend(y_auto)

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

    clf = _build_classifier(model_name)
    clf.fit(X, y_arr)

    artifact = {"model": clf, "feature_names": feature_names, "label_names": ROOM_RESET_ACTIONS, "model_name": model_name}
    joblib.dump(artifact, model_path)
    return model_path


def _build_classifier(model_name: str):
    key = str(model_name).strip().lower()
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=240,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=2,
        )
    if key == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    if key == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=42)
    raise ValueError(f"Unsupported model_name={model_name!r}. Choose from {MODEL_CHOICES}.")


def build_room_reset_training_frame(
    *,
    realtime_csv_path: str | Path,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    realtime_csv_path = Path(realtime_csv_path)
    cfg = load_hardware_config()
    sample_interval_sec = int(cfg.get("sample_interval_sec", 10))

    df = load_realtime_data(str(realtime_csv_path))
    session_root = repo_root() / "data" / "room_reset_sessions"

    X_rows: List[Dict[str, float]]
    y: List[str]
    X_rows, y = ([], [])

    X_human, y_human = _extract_training_from_human_labels(
        df,
        session_root=session_root,
        sample_interval_sec=sample_interval_sec,
    )
    if y_human:
        X_rows.extend(X_human)
        y.extend(y_human)

    if not force_synthetic:
        X_auto, y_auto = _extract_training_from_realtime(df, sample_interval_sec=sample_interval_sec)
        X_rows.extend(X_auto)
        y.extend(y_auto)

    min_samples = 250
    if len(y) < min_samples:
        X_syn, y_syn = _bootstrap_synthetic_training(sample_interval_sec=sample_interval_sec)
        X_rows = X_rows + X_syn
        y = y + y_syn

    if len(y) == 0:
        X_syn, y_syn = _bootstrap_synthetic_training(sample_interval_sec=sample_interval_sec, n_sequences=3, duration_minutes=120)
        X_rows = X_syn
        y = y_syn

    feature_names = sorted(X_rows[0].keys()) if X_rows else []
    X = pd.DataFrame([{k: row.get(k, 0.0) for k in feature_names} for row in X_rows], columns=feature_names)
    out = X.copy()
    out["label"] = np.array(y, dtype=object)
    return out


def evaluate_room_reset_model(
    *,
    realtime_csv_path: str | Path,
    model_name: str,
    test_fraction: float = 0.25,
    force_synthetic: bool = False,
) -> Dict[str, Any]:
    data = build_room_reset_training_frame(realtime_csv_path=realtime_csv_path, force_synthetic=force_synthetic)
    if data.empty or len(data) < 4:
        return {
            "model_name": model_name,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "n_total": int(len(data)),
            "n_train": 0,
            "n_test": 0,
            "confusion_matrix": [],
        }

    split_idx = int(len(data) * (1.0 - test_fraction))
    split_idx = max(1, min(split_idx, len(data) - 1))
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    clf = _build_classifier(model_name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=ROOM_RESET_ACTIONS)

    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "n_total": int(len(data)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    train_room_reset_model(
        realtime_csv_path=root / "data" / "realtime.csv",
        model_path=root / "models" / "room_reset" / "model.joblib",
    )

