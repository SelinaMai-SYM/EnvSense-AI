from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.room_reset.train import evaluate_room_reset_model, train_room_reset_model
from models.sleep_guard.train import evaluate_sleep_guard_model, train_sleep_guard_model
from utils.io_utils import repo_root


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("envsense_train")
MODEL_CANDIDATES = ["random_forest", "xgboost", "lightgbm"]


def _save_model_score_chart(df: pd.DataFrame, *, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.arange(len(df))
    width = 0.35
    acc = df["accuracy"].to_numpy(dtype=float) * 100.0
    macro = df["macro_f1"].to_numpy(dtype=float) * 100.0

    bars1 = ax.bar(x - width / 2, acc, width=width, color="#2563eb", label="Accuracy")
    bars2 = ax.bar(x + width / 2, macro, width=width, color="#f59e0b", label="Macro F1")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n") for name in df["model_name"]], fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_title(title, fontsize=12.5, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.0, f"{height:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(
    *,
    skip_realtime_labels: bool = False,
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
            skip_realtime_labels=skip_realtime_labels,
        )
        room_eval_rows.append(m)

    room_eval_df = pd.DataFrame(room_eval_rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_room_model = str(room_eval_df.iloc[0]["model_name"])
    logger.info("Room Reset best model: %s", best_room_model)
    train_room_reset_model(
        realtime_csv_path=room_csv,
        model_path=root / "models" / "room_reset" / "model.joblib",
        skip_realtime_labels=skip_realtime_labels,
        model_name=best_room_model,
    )

    logger.info("Evaluating candidate models for Sleep Guard...")
    sleep_eval_rows: List[Dict[str, object]] = []
    for model_name in MODEL_CANDIDATES:
        m = evaluate_sleep_guard_model(
            realtime_csv_path=sleep_csv,
            model_name=model_name,
            test_fraction=test_fraction,
            skip_realtime_labels=skip_realtime_labels,
        )
        sleep_eval_rows.append(m)

    sleep_eval_df = pd.DataFrame(sleep_eval_rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_sleep_model = str(sleep_eval_df.iloc[0]["model_name"])
    logger.info("Sleep Guard best model: %s", best_sleep_model)
    train_sleep_guard_model(
        realtime_csv_path=sleep_csv,
        model_path=root / "models" / "sleep_guard" / "model.joblib",
        skip_realtime_labels=skip_realtime_labels,
        model_name=best_sleep_model,
    )

    # Persist evaluation artifacts for reports and reproducibility
    report_dir = root / "models" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    room_eval_df.to_csv(report_dir / "room_reset_model_comparison.csv", index=False)
    sleep_eval_df.to_csv(report_dir / "sleep_guard_model_comparison.csv", index=False)
    figure_dir = root / "report_assets" / "figures"
    _save_model_score_chart(
        room_eval_df,
        title="Room Reset Coach",
        path=figure_dir / "room_reset_model_scores.png",
    )
    _save_model_score_chart(
        sleep_eval_df,
        title="Sleep Guard",
        path=figure_dir / "sleep_guard_model_scores.png",
    )
    summary = {
        "room_reset_best_model": best_room_model,
        "sleep_guard_best_model": best_sleep_model,
        "test_fraction": test_fraction,
        "skip_realtime_labels": skip_realtime_labels,
    }
    (report_dir / "best_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Model comparison tables saved under models/reports/")
    logger.info("All training jobs completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Room Reset and Sleep Guard models.")
    parser.add_argument(
        "--skip-realtime-labels",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--room-reset-csv",
        type=str,
        default=None,
        help="CSV path (relative to repo root) for Room Reset training, e.g. data/realtime_study.csv",
    )
    parser.add_argument(
        "--sleep-csv",
        type=str,
        default=None,
        help="CSV path (relative to repo root) for Sleep Guard training, e.g. data/realtime_sleep.csv",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Latest fraction used as time-based evaluation split for model selection.",
    )
    args = parser.parse_args()
    main(
        skip_realtime_labels=args.skip_realtime_labels,
        room_reset_csv_path=args.room_reset_csv,
        sleep_csv_path=args.sleep_csv,
        test_fraction=args.test_fraction,
    )

