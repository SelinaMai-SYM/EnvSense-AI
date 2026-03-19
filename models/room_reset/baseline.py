from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from features.window_features import compute_window_features


ROOM_STATE_OPTIONS = ["Good now", "Getting stuffy", "Poor ventilation", "Uncertain"]
BEST_ACTION_OPTIONS = ["Stay", "Open window", "Open door", "Move soon"]
CONFIDENCE_LABELS = ["High", "Medium", "Low"]


def _confidence_label_from_score(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.52:
        return "Medium"
    return "Low"


def baseline_room_reset_from_features(df_window: pd.DataFrame) -> Dict[str, str]:
    """
    Interpretable baseline logic for Room Reset Coach.

    Input df_window should already represent the past 5 minutes.
    """
    feats = compute_window_features(df_window)

    eco2_mean = feats["eco2_ppm_mean"]
    eco2_slope = feats["eco2_ppm_slope"]
    eco2_high_fraction = feats["eco2_high_fraction"]
    tvoc_spike = feats["tvoc_spike_flag"]
    humidity_out_frac = feats["humidity_out_of_range_fraction"]

    temp_last = feats["temp_C_last"]
    humidity_last = feats["humidity_last"]

    thermal_ok = (20.0 <= temp_last <= 26.0) and (35.0 <= humidity_last <= 55.0)

    # Strong ventilation needed: sustained high eCO2
    if eco2_high_fraction >= 0.60 or eco2_mean >= 1150.0:
        room_state = "Poor ventilation"
        best_action = "Open window"
        conf_score = 0.90 - 0.15 * float(max(0.0, eco2_mean - 1100.0) / 1200.0)
        explanation = f"eCO2 長期偏高（{eco2_mean:.0f} ppm，比例={eco2_high_fraction:.2f}），建议开窗快速降下来。"
        return {
            "room_state": room_state,
            "best_action": best_action,
            "confidence_label": _confidence_label_from_score(conf_score),
            "explanation": explanation,
        }

    # Eco2 rising indicates poor ventilation or occupancy load
    if eco2_slope > 6.0 and eco2_mean >= 900.0:
        room_state = "Getting stuffy"
        # If air already smells / VOC spikes, open door (cross-ventilation)
        best_action = "Open door" if tvoc_spike >= 1.0 else "Open window"
        conf_score = 0.68 + min(0.2, eco2_slope / 40.0)
        explanation = f"eCO2 正在上升（斜率≈{eco2_slope:.1f}），且当前空气不太稳定；建议先做通风。"
        return {
            "room_state": room_state,
            "best_action": best_action,
            "confidence_label": _confidence_label_from_score(conf_score),
            "explanation": explanation,
        }

    # TVOC spike without extreme eco2 often indicates air quality degradation
    if tvoc_spike >= 1.0 and eco2_mean >= 820.0:
        room_state = "Getting stuffy"
        best_action = "Open door"
        conf_score = 0.62
        explanation = "TVOC 出现尖峰，可能存在短时空气质量恶化；建议开门形成对流。"
        return {
            "room_state": room_state,
            "best_action": best_action,
            "confidence_label": _confidence_label_from_score(conf_score),
            "explanation": explanation,
        }

    # Thermal comfort issues with stable eco2: still could feel stuffy
    if not thermal_ok and humidity_out_frac > 0.15 and eco2_mean < 1050.0:
        room_state = "Getting stuffy"
        best_action = "Open window"
        conf_score = 0.55
        explanation = "温湿度有偏离（让人更不舒服），同时 eCO2 未明显恶化；开窗改善体感。"
        return {
            "room_state": room_state,
            "best_action": best_action,
            "confidence_label": _confidence_label_from_score(conf_score),
            "explanation": explanation,
        }

    # Comfortable + stable: stay
    if eco2_mean < 900.0 and abs(eco2_slope) <= 3.0 and tvoc_spike < 1.0 and thermal_ok and humidity_out_frac < 0.10:
        room_state = "Good now"
        best_action = "Stay"
        conf_score = 0.86
        explanation = "eCO2 处于较舒适区间且趋势平稳；温湿度也在舒适范围内，当前适合继续学习。"
        return {
            "room_state": room_state,
            "best_action": best_action,
            "confidence_label": _confidence_label_from_score(conf_score),
            "explanation": explanation,
        }

    # Default: uncertain
    room_state = "Uncertain"
    best_action = "Move soon"
    conf_score = 0.40
    explanation = "数据不足或指标冲突较多，无法做出高把握建议；倾向于先做一次通风/调整。"
    return {
        "room_state": room_state,
        "best_action": best_action,
        "confidence_label": _confidence_label_from_score(conf_score),
        "explanation": explanation,
    }

