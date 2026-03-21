from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from features.window_features import compute_window_features, get_recent_window, load_realtime_data
from models.sleep_guard.baseline import baseline_sleep_guard_from_features


CONFIDENCE_LABELS = ["High", "Medium", "Low"]


def _confidence_label_from_confidence(conf: float) -> str:
    if conf >= 0.75:
        return "High"
    if conf >= 0.52:
        return "Medium"
    return "Low"


def _map_readiness_to_action(readiness: str) -> str:
    mapping = {
        "Good to sleep": "Sleep now",
        "Sleep okay after ventilating": "Ventilate for 10–15 minutes",
        "Not ideal yet": "Keep door open",
    }
    return mapping.get(readiness, "Keep door open")


def predict_sleep_guard(
    *,
    csv_path: str | Path,
    minutes: int = 30,
    model_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Return:
      {
        "sleep_readiness": ...,
        "bedtime_action": ...,
        "main_risk_reason": ...,
        "confidence": float,
        "confidence_label": ...
      }
    """
    csv_path = str(csv_path)
    df = load_realtime_data(csv_path)
    df_window = get_recent_window(df, minutes=minutes)

    baseline = baseline_sleep_guard_from_features(df_window)
    default_conf = {"High": 0.86, "Medium": 0.63, "Low": 0.40}.get(
        baseline.get("confidence_label", "Low"), 0.40
    )

    features = compute_window_features(df_window)

    if model_path is None:
        # Default model lives next to this infer module.
        model_path = Path(__file__).resolve().parent / "model.joblib"
    model_path = Path(model_path)

    if not model_path.exists():
        return {
            "sleep_readiness": baseline["sleep_readiness"],
            "bedtime_action": baseline["bedtime_action"],
            "main_risk_reason": baseline["main_risk_reason"],
            "confidence": float(default_conf),
            "confidence_label": baseline["confidence_label"],
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
        class_index = int(np.argmax(proba[0]))
        conf = float(np.max(proba[0]))
        pred_readiness = clf.classes_[class_index]
        conf_label = _confidence_label_from_confidence(conf)

        action = _map_readiness_to_action(str(pred_readiness))
        # Keep baseline risk reason because it explains the current sensor situation.
        return {
            "sleep_readiness": str(pred_readiness),
            "bedtime_action": action,
            "main_risk_reason": baseline["main_risk_reason"],
            "confidence": conf,
            "confidence_label": conf_label,
        }
    except Exception:
        return {
            "sleep_readiness": baseline["sleep_readiness"],
            "bedtime_action": baseline["bedtime_action"],
            "main_risk_reason": baseline["main_risk_reason"],
            "confidence": float(default_conf),
            "confidence_label": baseline["confidence_label"],
        }

