from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from models.room_reset.train import evaluate_room_reset_model, train_room_reset_model
from models.sleep_guard.train import evaluate_sleep_guard_model, train_sleep_guard_model
from utils.io_utils import repo_root


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("envsense_train")
MODEL_CANDIDATES = ["random_forest", "gradient_boosting", "hist_gradient_boosting"]


def main(
    *,
    force_synthetic: bool = False,
    room_reset_csv_path: str | None = None,
    sleep_csv_path: str | None = None,
    test_fraction: float = 0.25,
) -> None:
    root = repo_root()
    room_csv = root / "data" / "realtime.csv" if room_reset_csv_path is None else (root / room_reset_csv_path)
    sleep_csv = root / "data" / "realtime.csv" if sleep_csv_path is None else (root / sleep_csv_path)

    logger.info("Evaluating candidate models for Room Reset...")
    room_eval_rows: List[Dict[str, object]] = []
    for model_name in MODEL_CANDIDATES:
        m = evaluate_room_reset_model(
            realtime_csv_path=room_csv,
            model_name=model_name,
            test_fraction=test_fraction,
            force_synthetic=force_synthetic,
        )
        room_eval_rows.append(m)

    room_eval_df = pd.DataFrame(room_eval_rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_room_model = str(room_eval_df.iloc[0]["model_name"])
    logger.info("Room Reset best model: %s", best_room_model)
    train_room_reset_model(
        realtime_csv_path=room_csv,
        model_path=root / "models" / "room_reset" / "model.joblib",
        force_synthetic=force_synthetic,
        model_name=best_room_model,
    )

    logger.info("Evaluating candidate models for Sleep Guard...")
    sleep_eval_rows: List[Dict[str, object]] = []
    for model_name in MODEL_CANDIDATES:
        m = evaluate_sleep_guard_model(
            realtime_csv_path=sleep_csv,
            model_name=model_name,
            test_fraction=test_fraction,
            force_synthetic=force_synthetic,
        )
        sleep_eval_rows.append(m)

    sleep_eval_df = pd.DataFrame(sleep_eval_rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_sleep_model = str(sleep_eval_df.iloc[0]["model_name"])
    logger.info("Sleep Guard best model: %s", best_sleep_model)
    train_sleep_guard_model(
        realtime_csv_path=sleep_csv,
        model_path=root / "models" / "sleep_guard" / "model.joblib",
        force_synthetic=force_synthetic,
        model_name=best_sleep_model,
    )

    # Persist evaluation artifacts for reports and reproducibility
    report_dir = root / "models" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    room_eval_df.to_csv(report_dir / "room_reset_model_comparison.csv", index=False)
    sleep_eval_df.to_csv(report_dir / "sleep_guard_model_comparison.csv", index=False)
    summary = {
        "room_reset_best_model": best_room_model,
        "sleep_guard_best_model": best_sleep_model,
        "test_fraction": test_fraction,
        "force_synthetic": force_synthetic,
    }
    (report_dir / "best_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Model comparison tables saved under models/reports/")
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
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Latest fraction used as time-based evaluation split for model selection.",
    )
    args = parser.parse_args()
    main(
        force_synthetic=args.force_synthetic,
        room_reset_csv_path=args.room_reset_csv,
        sleep_csv_path=args.sleep_csv,
        test_fraction=args.test_fraction,
    )

