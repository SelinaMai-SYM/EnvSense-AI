from __future__ import annotations

from typing import Dict

import pandas as pd

from features.window_features import compute_window_features


SLEEP_READINESS_OPTIONS = ["Good to sleep", "Sleep okay after ventilating", "Not ideal yet"]
BEDTIME_ACTION_OPTIONS = ["Sleep now", "Ventilate for 10–15 minutes", "Keep door open"]


def _confidence_label_from_score(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.52:
        return "Medium"
    return "Low"


def baseline_sleep_guard_from_features(df_window: pd.DataFrame) -> Dict[str, str]:
    """
    Interpretable baseline logic for Dorm Sleep Guard.

    Input df_window should already represent the past 30 minutes (default in app/infer).
    """
    feats = compute_window_features(df_window)

    eco2_mean = feats["eco2_ppm_mean"]
    eco2_max = feats["eco2_ppm_max"]
    eco2_slope = feats["eco2_ppm_slope"]
    eco2_high_fraction = feats["eco2_high_fraction"]

    temp_mean = feats["temp_C_mean"]
    humidity_mean = feats["humidity_mean"]
    temp_std = feats["temp_C_std"]
    humidity_std = feats["humidity_std"]
    humidity_out_frac = feats["humidity_out_of_range_fraction"]

    thermal_ok = (19.0 <= temp_mean <= 25.5) and (35.0 <= humidity_mean <= 55.0)
    thermal_stable = (temp_std <= 1.6) and (humidity_std <= 7.0) and humidity_out_frac <= 0.20

    # Core sleep readiness decision
    if eco2_mean <= 900.0 and eco2_max <= 1100.0 and eco2_slope <= 8.0 and thermal_ok and thermal_stable:
        sleep_readiness = "Good to sleep"
        bedtime_action = "Sleep now"
        risk = "eCO2 and thermal comfort are both stable."
        conf_score = 0.88 - 0.15 * float(max(0.0, eco2_high_fraction - 0.1) / 0.9)
    elif eco2_mean <= 1050.0 and eco2_max <= 1300.0 and (thermal_ok or thermal_stable) and eco2_slope <= 14.0:
        sleep_readiness = "Sleep okay after ventilating"
        bedtime_action = "Ventilate for 10–15 minutes"
        risk_bits = []
        if eco2_mean > 900.0:
            risk_bits.append(f"eCO2 is slightly elevated (avg~{eco2_mean:.0f})")
        if not thermal_ok:
            risk_bits.append("thermal comfort is slightly off")
        if not thermal_stable:
            risk_bits.append("temperature/humidity fluctuate noticeably")
        risk = "; ".join(risk_bits)[:80] if risk_bits else "Conditions are mildly suboptimal; ventilate first."
        conf_score = 0.62 + 0.12 * float(max(0.0, 1050.0 - eco2_mean) / 300.0)
    else:
        sleep_readiness = "Not ideal yet"
        bedtime_action = "Keep door open"
        risk_bits = []
        if eco2_mean > 1050.0:
            risk_bits.append(f"eCO2 has remained high for a while (avg~{eco2_mean:.0f})")
        if eco2_high_fraction > 0.4:
            risk_bits.append(f"high eCO2 ratio={eco2_high_fraction:.2f}")
        if not thermal_stable:
            risk_bits.append("temperature/humidity are unstable")
        if not risk_bits:
            risk_bits.append("ventilation conditions are not ideal")
        risk = "; ".join(risk_bits)[:100]
        conf_score = 0.48 + 0.35 * float(min(1.0, eco2_high_fraction + (1.0 if not thermal_stable else 0.0) * 0.25))

    return {
        "sleep_readiness": sleep_readiness,
        "bedtime_action": bedtime_action,
        "main_risk_reason": risk,
        "confidence_label": _confidence_label_from_score(conf_score),
    }

