from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import pandas as pd
import yaml


ModeName = Literal["study", "sleep"]
SENSOR_COLUMNS = {"timestamp", "temp_C", "humidity", "eco2_ppm", "tvoc"}
SLEEP_FEEDBACK_VALUES = {"slept_well", "okay", "poor_sleep"}


@dataclass(frozen=True)
class ModeDataPaths:
    sensor_csv: Optional[Path]
    annotation_table: Optional[Path]


@dataclass(frozen=True)
class ProjectDataPaths:
    study: ModeDataPaths
    sleep: ModeDataPaths
    feedback_output_root: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def load_hardware_config() -> Dict[str, Any]:
    """Load `config_hardware/config.yaml` if present, else fall back to example."""
    root = repo_root()
    config_path = root / "config_hardware" / "config.yaml"
    example_path = root / "config_hardware" / "config.example.yaml"

    cfg = load_yaml(config_path)
    if cfg:
        return cfg
    return load_yaml(example_path)


def _coerce_optional_path(value: Any, *, root: Path) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    p = Path(text)
    return p if p.is_absolute() else root / p


def _read_csv_preview(path: Path, *, nrows: int = 24) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return pd.DataFrame()


def _is_sensor_table(df_preview: pd.DataFrame) -> bool:
    return SENSOR_COLUMNS.issubset({str(col) for col in df_preview.columns})


def _find_study_annotation_column(df_preview: pd.DataFrame) -> Optional[str]:
    columns = {str(col) for col in df_preview.columns}
    if "timestamp" not in columns:
        return None
    for candidate in ("best_action", "action", "label"):
        if candidate in columns:
            return candidate
    return None


def _find_sleep_feedback_column(df_preview: pd.DataFrame) -> Optional[str]:
    columns = [str(col) for col in df_preview.columns]
    if "timestamp" not in columns:
        return None
    for column in columns:
        if column == "timestamp":
            continue
        sample = (
            df_preview[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
        if sample and set(sample).issubset(SLEEP_FEEDBACK_VALUES):
            return column
    return None


def _sensor_priority(sensor_path: Path) -> tuple[float, float, int, str]:
    preview = _read_csv_preview(sensor_path, nrows=240)
    if preview.empty or not _is_sensor_table(preview):
        return (-1.0, 0.0, 0, sensor_path.name.lower())
    eco2_mean = float(pd.to_numeric(preview["eco2_ppm"], errors="coerce").mean())
    temp_mean = float(pd.to_numeric(preview["temp_C"], errors="coerce").mean())
    return (eco2_mean, -temp_mean, len(preview), sensor_path.name.lower())


def _iter_csv_paths(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    return sorted(p for p in data_root.rglob("*.csv") if p.is_file())


@lru_cache(maxsize=1)
def discover_project_data_paths() -> ProjectDataPaths:
    root = repo_root()
    cfg = load_hardware_config()
    data_cfg = cfg.get("data_paths", {}) if isinstance(cfg.get("data_paths"), dict) else {}
    data_root = root / "data"

    sensor_by_dir: dict[Path, list[Path]] = {}
    study_tables: list[Path] = []
    sleep_tables: list[Path] = []

    for csv_path in _iter_csv_paths(data_root):
        preview = _read_csv_preview(csv_path)
        if preview.empty:
            continue
        if _is_sensor_table(preview):
            sensor_by_dir.setdefault(csv_path.parent, []).append(csv_path)
            continue
        if _find_study_annotation_column(preview):
            study_tables.append(csv_path)
            continue
        if _find_sleep_feedback_column(preview):
            sleep_tables.append(csv_path)

    def choose_mode_paths(mode: ModeName, discovered_tables: list[Path]) -> ModeDataPaths:
        sensor_key = f"{mode}_offline_csv"
        table_key = "study_annotation_table" if mode == "study" else "sleep_feedback_table"

        sensor_csv = _coerce_optional_path(data_cfg.get(sensor_key), root=root)
        annotation_table = _coerce_optional_path(data_cfg.get(table_key), root=root)
        if sensor_csv and annotation_table:
            return ModeDataPaths(sensor_csv=sensor_csv, annotation_table=annotation_table)

        best_pair: tuple[tuple[float, float, int, str], Optional[Path], Path] | None = None
        for table_path in discovered_tables:
            candidates = sensor_by_dir.get(table_path.parent, [])
            sensor_candidate = max(candidates, key=_sensor_priority, default=None)
            pair_score = _sensor_priority(sensor_candidate) if sensor_candidate else (-1.0, 0.0, 0, "")
            if best_pair is None or pair_score > best_pair[0]:
                best_pair = (pair_score, sensor_candidate, table_path)

        if sensor_csv is None and best_pair is not None:
            sensor_csv = best_pair[1]
        if annotation_table is None and best_pair is not None:
            annotation_table = best_pair[2]
        return ModeDataPaths(sensor_csv=sensor_csv, annotation_table=annotation_table)

    feedback_output_root = _coerce_optional_path(data_cfg.get("feedback_output_root"), root=root)
    if feedback_output_root is None:
        feedback_output_root = root / "data" / "session_feedback"

    return ProjectDataPaths(
        study=choose_mode_paths("study", study_tables),
        sleep=choose_mode_paths("sleep", sleep_tables),
        feedback_output_root=feedback_output_root,
    )


def resolve_mode_data_paths(mode: ModeName) -> ModeDataPaths:
    paths = discover_project_data_paths()
    return paths.study if mode == "study" else paths.sleep


def resolve_feedback_output_root() -> Path:
    return discover_project_data_paths().feedback_output_root


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_nested(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

