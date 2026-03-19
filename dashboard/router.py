from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from features.window_features import get_recent_window, load_realtime_data
from models.room_reset.infer import predict_room_reset
from models.sleep_guard.infer import predict_sleep_guard
from utils.io_utils import repo_root


def set_mode_file(mode: str) -> None:
    root = repo_root()
    mode_file = root / "data" / "last_mode.txt"
    try:
        mode_file.parent.mkdir(parents=True, exist_ok=True)
        mode_file.write_text(mode, encoding="utf-8")
    except Exception:
        return


def run_pipeline(*, mode: str, csv_path: str | Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Returns:
      (inference_result_dict, df_window_for_plots)
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"study", "sleep"}:
        mode_norm = "study"

    df = load_realtime_data(str(csv_path))
    if mode_norm == "sleep":
        result = predict_sleep_guard(csv_path=csv_path, minutes=30)
        df_window = get_recent_window(df, minutes=30)
    else:
        result = predict_room_reset(csv_path=csv_path, minutes=5)
        df_window = get_recent_window(df, minutes=5)

    set_mode_file(mode_norm)
    return result, df_window

