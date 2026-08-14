"""配置加载 — 全项目读 configs/phased_array.yaml 的统一入口。"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phased_array.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """安全加载 YAML 配置。path 为 None 时读 configs/phased_array.yaml。"""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"空配置文件: {cfg_path}")
    return cfg
