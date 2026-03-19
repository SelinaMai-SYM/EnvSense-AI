from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .env-card {
          border: 1px solid rgba(0,0,0,0.08);
          border-radius: 14px;
          padding: 14px;
          background: rgba(255,255,255,0.75);
        }
        .env-title {
          font-weight: 700;
          margin-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_latest_row(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Timestamp]]:
    if df is None or df.empty:
        return None, None
    df = df.sort_values("timestamp")
    last = df.iloc[-1]
    return last, last["timestamp"]


def render_sensor_panel(latest: Optional[pd.Series], last_updated: Optional[pd.Timestamp]) -> None:
    with st.container():
        st.markdown('<div class="env-card">', unsafe_allow_html=True)
        st.markdown('<div class="env-title">Live Sensor Panel</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        if latest is None:
            c1.metric("Temperature (°C)", "N/A")
            c2.metric("Humidity (%)", "N/A")
            c3.metric("eCO2 (ppm)", "N/A")
            c4.metric("TVOC", "N/A")
        else:
            c1.metric("Temperature (°C)", f"{float(latest['temp_C']):.1f}")
            c2.metric("Humidity (%)", f"{float(latest['humidity']):.0f}")
            c3.metric("eCO2 (ppm)", f"{float(latest['eco2_ppm']):.0f}")
            c4.metric("TVOC", f"{float(latest['tvoc']):.0f}")

        if last_updated is None:
            st.caption("Last updated: N/A")
        else:
            ts_str = str(pd.to_datetime(last_updated, utc=True).isoformat())
            st.caption(f"Last updated: {ts_str}")

        st.markdown("</div>", unsafe_allow_html=True)


def render_output_card(title: str, value: str, *, subtitle: Optional[str] = None) -> None:
    with st.container():
        st.markdown('<div class="env-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="env-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f"**{value}**")
        if subtitle:
            st.caption(subtitle)
        st.markdown("</div>", unsafe_allow_html=True)


def render_confidence_card(confidence: float, confidence_label: str) -> None:
    with st.container():
        st.markdown('<div class="env-card">', unsafe_allow_html=True)
        st.markdown('<div class="env-title">Confidence</div>', unsafe_allow_html=True)
        st.markdown(f"**{confidence:.2f}**  ")
        st.caption(f"Label: {confidence_label}")
        st.markdown("</div>", unsafe_allow_html=True)


def plot_trend(df_window: pd.DataFrame, column: str, title: str) -> None:
    if df_window is None or df_window.empty or column not in df_window.columns:
        st.info(f"No data for {title} yet.")
        return

    df_window = df_window.sort_values("timestamp")
    x = pd.to_datetime(df_window["timestamp"], utc=True).dt.to_pydatetime()
    y = pd.to_numeric(df_window[column], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.plot(x, y, linewidth=1.8)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.margins(x=0)
    st.pyplot(fig, clear_figure=True)

