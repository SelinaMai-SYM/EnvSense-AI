from __future__ import annotations

import argparse
import logging

from models.room_reset.train import train_room_reset_model
from models.sleep_guard.train import train_sleep_guard_model
from utils.io_utils import repo_root


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("envsense_train")


def main(
    *,
    force_synthetic: bool = False,
    room_reset_csv_path: str | None = None,
    sleep_csv_path: str | None = None,
) -> None:
    root = repo_root()
    room_csv = root / "data" / "realtime.csv" if room_reset_csv_path is None else (root / room_reset_csv_path)
    sleep_csv = root / "data" / "realtime.csv" if sleep_csv_path is None else (root / sleep_csv_path)

    logger.info("Training Room Reset Coach model...")
    model_path_room = root / "models" / "room_reset" / "model.joblib"
    train_room_reset_model(
        realtime_csv_path=room_csv,
        model_path=model_path_room,
        force_synthetic=force_synthetic,
    )

    logger.info("Training Dorm Sleep Guard model...")
    model_path_sleep = root / "models" / "sleep_guard" / "model.joblib"
    train_sleep_guard_model(
        realtime_csv_path=sleep_csv,
        model_path=model_path_sleep,
        force_synthetic=force_synthetic,
    )

    logger.info("All training jobs completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Room Reset and Sleep Guard models.")
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Ignore realtime/session labels and train from synthetic data only.",
    )
    parser.add_argument(
        "--room-reset-csv",
        type=str,
        default=None,
        help="CSV path (relative to repo root) for Room Reset training, e.g. data/realtime_classroom.csv",
    )
    parser.add_argument(
        "--sleep-csv",
        type=str,
        default=None,
        help="CSV path (relative to repo root) for Sleep Guard training, e.g. data/realtime_bedroom.csv",
    )
    args = parser.parse_args()
    main(
        force_synthetic=args.force_synthetic,
        room_reset_csv_path=args.room_reset_csv,
        sleep_csv_path=args.sleep_csv,
    )

