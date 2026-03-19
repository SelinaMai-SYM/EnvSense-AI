from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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

