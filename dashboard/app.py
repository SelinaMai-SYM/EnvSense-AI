from __future__ import annotations

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard.components import ensure_mock_realtime_data, inject_styles, get_latest_row, render_sensor_panel
from dashboard.room_reset_page import render_room_reset_page
from dashboard.sleep_guard_page import render_sleep_guard_page
from features.window_features import load_realtime_data
from utils.io_utils import load_hardware_config, repo_root


def main() -> None:
    st.set_page_config(page_title="EnvSense AI", layout="wide")
    inject_styles()

    st_autorefresh(interval=10_000, limit=None, key="envsense_autorefresh")

    cfg = load_hardware_config()
    mock_mode = bool(cfg.get("mock_mode", True))
    sample_interval_sec = int(cfg.get("sample_interval_sec", 10))

    root = repo_root()
    csv_path = root / "data" / "realtime.csv"

    ensure_mock_realtime_data(
        csv_path=csv_path,
        mock_mode=mock_mode,
        sample_interval_sec=sample_interval_sec,
        minutes=60,
    )

    df = load_realtime_data(str(csv_path))
    latest, last_updated = get_latest_row(df)

    st.title("EnvSense AI")
    st.caption("Personal environment assistant for study and sleep")

    # Mode selection (home)
    if "mode" not in st.session_state:
        st.session_state.mode = None

    c_mode_1, c_mode_2 = st.columns(2)
    with c_mode_1:
        if st.button("Study Mode", use_container_width=True):
            st.session_state.mode = "study"
    with c_mode_2:
        if st.button("Sleep Mode", use_container_width=True):
            st.session_state.mode = "sleep"

    # Shared live panel always
    render_sensor_panel(latest, last_updated)

    st.divider()

    if st.session_state.mode == "study":
        render_room_reset_page(csv_path=csv_path)
    elif st.session_state.mode == "sleep":
        render_sleep_guard_page(csv_path=csv_path)
    else:
        st.info("Select a mode to see guidance.")


if __name__ == "__main__":
    main()

