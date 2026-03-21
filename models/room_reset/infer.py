from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from features.window_features import compute_window_features, get_recent_window, load_realtime_data
from models.room_reset.baseline import baseline_room_reset_from_features


CONFIDENCE_LABELS = ["High", "Medium", "Low"]


def _confidence_label_from_confidence(conf: float) -> str:
    if conf >= 0.75:
        return "High"
    if conf >= 0.52:
        return "Medium"
    return "Low"


def predict_room_reset(
    *,
    csv_path: str | Path,
    minutes: int = 5,
    model_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Return:
      {
        "room_state": ...,
        "best_action": ...,
        "confidence": float,
        "confidence_label": ...,
        "explanation": ...
      }
    """
    csv_path = str(csv_path)
    df = load_realtime_data(csv_path)
    df_window = get_recent_window(df, minutes=minutes)

    baseline = baseline_room_reset_from_features(df_window)

    default_conf = {"High": 0.85, "Medium": 0.62, "Low": 0.40}.get(
        baseline.get("confidence_label", "Low"), 0.40
    )

    features = compute_window_features(df_window)

    if model_path is None:
        # Default model lives next to this infer module.
        model_path = Path(__file__).resolve().parent / "model.joblib"

    model_path = Path(model_path)
    if not model_path.exists():
        return {
            "room_state": baseline["room_state"],
            "best_action": baseline["best_action"],
            "confidence": float(default_conf),
            "confidence_label": baseline["confidence_label"],
            "explanation": baseline["explanation"],
        }

    try:
        try:
            import joblib  # type: ignore
        except Exception:
            raise RuntimeError("joblib unavailable")

        artifact = joblib.load(model_path)
        clf = artifact["model"]
        feature_names = artifact.get("feature_names") or list(features.keys())

        X = pd.DataFrame([{k: features.get(k, 0.0) for k in feature_names}])
        proba = clf.predict_proba(X)
        if proba.size == 0:
            raise RuntimeError("empty proba")

        class_index = int(np.argmax(proba[0]))
        conf = float(np.max(proba[0]))
        pred_action = clf.classes_[class_index]

        conf_label = _confidence_label_from_confidence(conf)

        # Keep baseline room_state and add ML action selection
        eco2_mean = features.get("eco2_ppm_mean", 0.0)
        tvoc_spike = int(features.get("tvoc_spike_flag", 0.0))
        eco2_high_frac = features.get("eco2_high_fraction", 0.0)

        reason_bits = []
        reason_bits.append(f"eCO2≈{eco2_mean:.0f}ppm")
        if tvoc_spike:
            reason_bits.append("TVOC spike")
        if eco2_high_frac >= 0.4:
            reason_bits.append(f"high eCO2 ratio={eco2_high_frac:.2f}")

        action_explain = baseline["explanation"]
        if pred_action != baseline["best_action"]:
            action_explain = f"ML suggested action: {pred_action}. {action_explain}"

        return {
            "room_state": baseline["room_state"],
            "best_action": str(pred_action),
            "confidence": conf,
            "confidence_label": conf_label,
            "explanation": (f"{'; '.join(reason_bits)}. {action_explain}")[:180],
        }
    except Exception:
        # Fail gracefully to baseline
        return {
            "room_state": baseline["room_state"],
            "best_action": baseline["best_action"],
            "confidence": float(default_conf),
            "confidence_label": baseline["confidence_label"],
            "explanation": baseline["explanation"],
        }

