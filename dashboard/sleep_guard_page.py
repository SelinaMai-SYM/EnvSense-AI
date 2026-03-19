from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from dashboard.components import plot_trend, render_confidence_card, render_output_card
from dashboard.router import run_pipeline


def render_sleep_guard_page(*, csv_path: str | Path) -> None:
    st.markdown("## Dorm Sleep Guard (Sleep Mode)")
    result, df_window = run_pipeline(mode="sleep", csv_path=csv_path)

    c1, c2 = st.columns(2)
    with c1:
        render_output_card("Sleep readiness", str(result.get("sleep_readiness", "N/A")))
    with c2:
        render_output_card("Best bedtime action", str(result.get("bedtime_action", "N/A")))

    c3, c4 = st.columns(2)
    with c3:
        conf = float(result.get("confidence", 0.0))
        render_confidence_card(conf, str(result.get("confidence_label", "Low")))
    with c4:
        render_output_card("Main risk reason", str(result.get("main_risk_reason", ""))[:200])

    st.markdown("### Trends (past 30 minutes)")
    colA, colB = st.columns(2)
    with colA:
        plot_trend(df_window, "temp_C", "Temperature (°C)")
        plot_trend(df_window, "humidity", "Humidity (%)")
    with colB:
        plot_trend(df_window, "eco2_ppm", "eCO2 (ppm)")
        plot_trend(df_window, "tvoc", "TVOC")

    st.markdown("### Morning Check-in (placeholder)")
    st.caption("Future expansion: collect feedback to improve sleep-readiness model.")

    if "morning_checkin" not in st.session_state:
        st.session_state.morning_checkin = None

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Slept well", use_container_width=True):
            st.session_state.morning_checkin = "slept_well"
    with c2:
        if st.button("Okay", use_container_width=True):
            st.session_state.morning_checkin = "okay"
    with c3:
        if st.button("Poor sleep", use_container_width=True):
            st.session_state.morning_checkin = "poor_sleep"

    if st.session_state.morning_checkin:
        st.success(f"Recorded (not persisted yet): {st.session_state.morning_checkin}")

