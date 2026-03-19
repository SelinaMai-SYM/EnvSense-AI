from __future__ import annotations

import logging
from pathlib import Path

from models.room_reset.train import train_room_reset_model
from models.sleep_guard.train import train_sleep_guard_model
from utils.io_utils import repo_root


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("envsense_train")


def main() -> None:
    root = repo_root()
    csv_path = root / "data" / "realtime.csv"

    logger.info("Training Room Reset Coach model...")
    model_path_room = root / "models" / "room_reset" / "model.joblib"
    train_room_reset_model(realtime_csv_path=csv_path, model_path=model_path_room)

    logger.info("Training Dorm Sleep Guard model...")
    model_path_sleep = root / "models" / "sleep_guard" / "model.joblib"
    train_sleep_guard_model(realtime_csv_path=csv_path, model_path=model_path_sleep)

    logger.info("All training jobs completed.")


if __name__ == "__main__":
    main()

