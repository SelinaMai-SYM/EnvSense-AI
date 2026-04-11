from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard.components import plot_trend, render_confidence_card, render_output_card
from dashboard.router import run_pipeline


def render_room_reset_page(*, csv_path: str | Path) -> None:
    st.markdown("## Room Reset Coach (Study Mode)")
    result, df_window = run_pipeline(mode="study", csv_path=csv_path)

    c1, c2 = st.columns(2)
    with c1:
        render_output_card("Current room state", str(result.get("room_state", "N/A")))
    with c2:
        render_output_card("Best next action", str(result.get("best_action", "N/A")))

    c3, c4 = st.columns(2)
    with c3:
        conf = float(result.get("confidence", 0.0))
        render_confidence_card(conf, str(result.get("confidence_label", "Low")))
    with c4:
        render_output_card("Short explanation", str(result.get("explanation", ""))[:160])

    st.markdown("### Trends (past 5 minutes)")
    colA, colB = st.columns(2)
    with colA:
        plot_trend(df_window, "temp_C", "Temperature (°C)")
        plot_trend(df_window, "humidity", "Humidity (%)")
    with colB:
        plot_trend(df_window, "eco2_ppm", "eCO2 (ppm)")
        plot_trend(df_window, "tvoc", "TVOC")

