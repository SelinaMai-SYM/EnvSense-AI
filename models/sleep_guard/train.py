from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from features.labels import (
    SLEEP_READINESS,
    derive_sleep_readiness,
    load_sleep_feedback_labels,
    simulate_environment_series,
)
from features.window_features import compute_window_features, load_realtime_data
from models.label_encoded_classifier import LabelEncodedClassifier
from utils.io_utils import load_hardware_config, resolve_mode_data_paths

MODEL_CHOICES = ("random_forest", "xgboost", "lightgbm")


def _split_contiguous_segments(
    df: pd.DataFrame,
    *,
    sample_interval_sec: int,
    gap_multiplier: int = 6,
) -> List[pd.DataFrame]:
    """
    Split a realtime series into contiguous segments so multi-night CSVs do not
    produce windows that jump across long gaps between sessions.
    """
    if df.empty or "timestamp" not in df.columns:
        return []

    df_sorted = df.sort_values("timestamp").reset_index(drop=True).copy()
    df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], utc=True, errors="coerce")
    df_sorted = df_sorted.dropna(subset=["timestamp"])
    if df_sorted.empty:
        return []

    gap_threshold = pd.Timedelta(seconds=max(sample_interval_sec * gap_multiplier, sample_interval_sec + 1))
    segment_ids = (df_sorted["timestamp"].diff() > gap_threshold).cumsum()

    segments: List[pd.DataFrame] = []
    for _, segment in df_sorted.groupby(segment_ids, sort=False):
        segment = segment.reset_index(drop=True)
        if not segment.empty:
            segments.append(segment)
    return segments


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

    segments = _split_contiguous_segments(df, sample_interval_sec=sample_interval_sec)
    for segment in segments:
        if len(segment) < window_steps + 5:
            continue
        for idx in range(window_steps, len(segment), stride):
            df_window = segment.iloc[idx - window_steps : idx]
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


def _extract_training_from_human_labels(
    df: pd.DataFrame,
    *,
    annotation_sources: Iterable[str | Path],
    sample_interval_sec: int,
    window_minutes: int = 30,
) -> Tuple[List[Dict[str, float]], List[str]]:
    window_steps = max(2, int(window_minutes * 60 // sample_interval_sec))
    X_rows: List[Dict[str, float]] = []
    y: List[str] = []

    if df.empty:
        return X_rows, y

    segments = _split_contiguous_segments(df, sample_interval_sec=sample_interval_sec)
    if not segments:
        return X_rows, y

    gap_tolerance = pd.Timedelta(seconds=max(sample_interval_sec * 6, sample_interval_sec + 1))
    segment_windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = [
        (segment["timestamp"].iloc[0], segment["timestamp"].iloc[-1], segment) for segment in segments
    ]

    for source in annotation_sources:
        labels_df = load_sleep_feedback_labels(source)
        if labels_df.empty:
            continue

        labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"], utc=True, errors="coerce")
        labels_df = labels_df.dropna(subset=["timestamp", "sleep_readiness"])
        labels_df = labels_df[labels_df["sleep_readiness"].isin(SLEEP_READINESS)]
        if labels_df.empty:
            continue

        for _, row in labels_df.iterrows():
            ts = row["timestamp"]
            matched_segment: pd.DataFrame | None = None
            for seg_start, seg_end, segment in segment_windows:
                if seg_start <= ts <= seg_end + gap_tolerance:
                    matched_segment = segment
                    break

            if matched_segment is None:
                continue

            idx = int(matched_segment["timestamp"].searchsorted(ts, side="right"))
            if idx < window_steps:
                continue
            df_window = matched_segment.iloc[idx - window_steps : idx]
            if len(df_window) < window_steps:
                continue
            X_rows.append(compute_window_features(df_window))
            y.append(str(row["sleep_readiness"]))

    return X_rows, y


def train_sleep_guard_model(
    *,
    realtime_csv_path: str | Path,
    model_path: str | Path,
    force_synthetic: bool = False,
    model_name: str = "random_forest",
) -> Path:
    """
    Train a lightweight RandomForest model for Sleep Guard.
    """
    realtime_csv_path = Path(realtime_csv_path)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_hardware_config()
    sample_interval_sec = int(cfg.get("sample_interval_sec", 10))

    df = load_realtime_data(str(realtime_csv_path))
    mode_paths = resolve_mode_data_paths("sleep")
    annotation_sources = [mode_paths.annotation_table] if mode_paths.annotation_table else []

    X_rows, y = _extract_training_from_human_labels(
        df,
        annotation_sources=annotation_sources,
        sample_interval_sec=sample_interval_sec,
    )

    if not force_synthetic:
        X_auto, y_auto = _extract_training_from_realtime(df, sample_interval_sec=sample_interval_sec)
        X_rows.extend(X_auto)
        y.extend(y_auto)

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

    clf = _build_classifier(model_name, class_names=SLEEP_READINESS)
    clf.fit(X, y_arr)

    artifact = {"model": clf, "feature_names": feature_names, "label_names": SLEEP_READINESS, "model_name": model_name}
    joblib.dump(artifact, model_path)
    return model_path


def _build_classifier(model_name: str, *, class_names: List[str]):
    key = str(model_name).strip().lower()
    if key == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=320,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_samples_leaf=2,
        )
        return LabelEncodedClassifier(estimator, class_names=class_names)
    if key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is required for model_name='xgboost'.") from exc
        estimator = XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=len(class_names),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        return LabelEncodedClassifier(estimator, class_names=class_names)
    if key == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("lightgbm is required for model_name='lightgbm'.") from exc
        estimator = LGBMClassifier(
            n_estimators=240,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multiclass",
            num_class=len(class_names),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        return LabelEncodedClassifier(estimator, class_names=class_names)
    raise ValueError(f"Unsupported model_name={model_name!r}. Choose from {MODEL_CHOICES}.")


def build_sleep_guard_training_frame(
    *,
    realtime_csv_path: str | Path,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    realtime_csv_path = Path(realtime_csv_path)
    cfg = load_hardware_config()
    sample_interval_sec = int(cfg.get("sample_interval_sec", 10))

    df = load_realtime_data(str(realtime_csv_path))
    mode_paths = resolve_mode_data_paths("sleep")
    annotation_sources = [mode_paths.annotation_table] if mode_paths.annotation_table else []

    X_rows, y = _extract_training_from_human_labels(
        df,
        annotation_sources=annotation_sources,
        sample_interval_sec=sample_interval_sec,
    )

    if not force_synthetic:
        X_auto, y_auto = _extract_training_from_realtime(df, sample_interval_sec=sample_interval_sec)
        X_rows.extend(X_auto)
        y.extend(y_auto)

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
    out = X.copy()
    out["label"] = np.array(y, dtype=object)
    return out


def evaluate_sleep_guard_model(
    *,
    realtime_csv_path: str | Path,
    model_name: str,
    test_fraction: float = 0.25,
    force_synthetic: bool = False,
) -> Dict[str, Any]:
    data = build_sleep_guard_training_frame(realtime_csv_path=realtime_csv_path, force_synthetic=force_synthetic)
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

    clf = _build_classifier(model_name, class_names=SLEEP_READINESS)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=SLEEP_READINESS)

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
    train_sleep_guard_model(
        realtime_csv_path=root / "data" / "realtime.csv",
        model_path=root / "models" / "sleep_guard" / "model.joblib",
    )

