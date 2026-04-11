from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config_hardware.dht22 import DHT22Reader
from config_hardware.ens160 import ENS160Reader
from config_hardware.oled import OLEDDisplay
from features.window_features import get_recent_window, load_realtime_data
from models.room_reset.baseline import baseline_room_reset_from_features
from models.sleep_guard.baseline import baseline_sleep_guard_from_features
from utils.io_utils import load_hardware_config, repo_root
from utils.time_utils import now_iso


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("envsense")


REQUIRED_CSV_COLUMNS = ["timestamp", "temp_C", "humidity", "eco2_ppm", "tvoc"]


def ensure_realtime_csv(csv_path: Path) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_CSV_COLUMNS)


def append_realtime_row(csv_path: Path, row: Dict[str, float | str]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in REQUIRED_CSV_COLUMNS])


def read_last_mode(mode_file: Path) -> str:
    try:
        if not mode_file.exists():
            return "study"
        mode = mode_file.read_text(encoding="utf-8").strip().lower()
        if mode in {"study", "sleep"}:
            return mode
        return "study"
    except Exception:
        return "study"


def compute_mode_result_text(df: pd.DataFrame, mode: str) -> str:
    """
    Provide a short text for OLED. Keep it lightweight (baseline only).
    """
    if mode == "sleep":
        df_window = get_recent_window(df, minutes=30)
        features = baseline_sleep_guard_from_features(df_window)
        action = features.get("bedtime_action", "Ventilate")
        risk = features.get("main_risk_reason", "").strip()
        return f"Sleep: {action}\n{risk}"[:32]

    df_window = get_recent_window(df, minutes=5)
    features = baseline_room_reset_from_features(df_window)
    action = features.get("best_action", "Stay")
    explanation = features.get("explanation", "")
    return f"Study: {action}\n{explanation}"[:32]


def main(*, iterations: Optional[int] = None, csv_path_override: Optional[str] = None) -> None:
    cfg = load_hardware_config()
    interval_sec = int(cfg.get("sample_interval_sec", 10))

    root = repo_root()
    csv_path = Path(csv_path_override) if csv_path_override else (root / "data" / "realtime.csv")
    mode_file = root / "data" / "last_mode.txt"

    dht_reader = DHT22Reader(cfg=cfg.get("dht22", {}))
    ens_reader = ENS160Reader(cfg=cfg.get("ens160", {}))

    oled_cfg = cfg.get("oled", {}) or {}
    oled_enabled = bool(oled_cfg.get("enabled", False))
    oled = OLEDDisplay(oled_cfg, enabled=oled_enabled)

    ensure_realtime_csv(csv_path)

    logger.info("Starting EnvSense-AI live logger (interval=%ss)", interval_sec)

    i = 0
    oled_every = 6  # ~1 minute with interval=10s
    while True:
        if iterations is not None and i >= iterations:
            logger.info("Reached iterations=%d, stopping.", iterations)
            break

        start = time.time()
        ts = now_iso()

        dht_data = dht_reader.read()
        ens_data = ens_reader.read()

        row = {
            "timestamp": ts,
            "temp_C": dht_data["temp_C"],
            "humidity": dht_data["humidity"],
            "eco2_ppm": ens_data["eco2_ppm"],
            "tvoc": ens_data["tvoc"],
        }

        append_realtime_row(csv_path, row)
        logger.debug("Logged: %s", row)

        if i % oled_every == 0:
            try:
                last_mode = read_last_mode(mode_file)
                if oled.enabled:
                    df = load_realtime_data(csv_path)
                    result_text = compute_mode_result_text(df, last_mode)
                    oled.show(
                        values={"temp_C": row["temp_C"], "humidity": row["humidity"], "eco2_ppm": row["eco2_ppm"], "tvoc": row["tvoc"]},
                        mode_result_text=result_text,
                    )
            except Exception:
                # Never crash the logger loop due to OLED or inference issues
                pass

        i += 1
        elapsed = time.time() - start
        time.sleep(max(0.0, interval_sec - elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnvSense-AI sensor logger")
    parser.add_argument("--iterations", type=int, default=None, help="Run a finite number of loops (debug).")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional output CSV path for collected rows.",
    )
    args = parser.parse_args()
    main(iterations=args.iterations, csv_path_override=args.csv_path)

