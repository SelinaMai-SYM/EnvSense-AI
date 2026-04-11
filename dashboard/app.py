from __future__ import annotations

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard.components import inject_styles, get_latest_row, render_sensor_panel
from dashboard.room_reset_page import render_room_reset_page
from dashboard.sleep_guard_page import render_sleep_guard_page
from features.window_features import load_realtime_data
from utils.io_utils import repo_root, resolve_mode_data_paths


def main() -> None:
    st.set_page_config(page_title="EnvSense AI", layout="wide")
    inject_styles()

    root = repo_root()

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

    data_source = st.sidebar.radio("Data Source", ["Live Input", "Offline Example"], index=0)

    mode = st.session_state.mode
    mode_for_path = mode if mode in {"study", "sleep"} else "study"

    if data_source == "Live Input":
        csv_path = root / "data" / "realtime.csv"
        st_autorefresh(interval=10_000, limit=None, key="envsense_autorefresh")
    else:
        mode_paths = resolve_mode_data_paths(mode_for_path)
        csv_path = mode_paths.sensor_csv or (root / "data" / "realtime.csv")

    st.caption("Current source: live stream" if data_source == "Live Input" else "Current source: offline example")

    df = load_realtime_data(str(csv_path))
    latest, last_updated = get_latest_row(df)

    # Shared live panel always
    render_sensor_panel(latest, last_updated)
    if latest is None:
        if data_source == "Live Input":
            st.warning("还没有实时数据。请先运行 `python3 main.py` 并写入 `data/realtime.csv`。")
        else:
            st.warning("离线示例数据为空。请检查本地离线数据配置。")

    st.divider()

    if mode == "study":
        render_room_reset_page(csv_path=csv_path)
    elif mode == "sleep":
        render_sleep_guard_page(csv_path=csv_path)
    else:
        st.info("Select a mode to see guidance.")


if __name__ == "__main__":
    main()

