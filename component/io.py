# -*- coding: utf-8 -*-
"""component.predict 的输入/输出隔离层。

铁律（契约 §9 遥测与标签分离）：
- 监督信号（RUL / 真值退化状态等）绝不进入推理遥测；
  列级命中 FORBIDDEN_INFERENCE_COLUMNS 即拒绝；
  值级（长表 telemetry_name 取值命中，如 true_rul / damage_truth）同样拒绝。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

# 列级禁入（宽表遗留列名 + 契约标签表字段）
FORBIDDEN_INFERENCE_COLUMNS = {
    "rul", "true_rul", "label", "degradation_state", "event_observed",
    "rul_lower_bound", "damage_truth", "damage", "label_fail",
}
# 值级禁入：长表 telemetry_name 不得出现任何监督/真值通道
FORBIDDEN_TELEMETRY_NAMES = FORBIDDEN_INFERENCE_COLUMNS | {
    "hi_array", "hi_ch", "z_ch", "rul_ch", "soh",
}


@dataclass(frozen=True)
class PredictionRequest:
    telemetry: pd.DataFrame
    metadata: dict
    telemetry_name: str


def load_request(telemetry_path: Path, metadata_path: Path,
                 requested_name: str | None) -> PredictionRequest:
    """读入遥测长表 + 元数据，完成标签隔离与主遥测选取。"""
    frame = pd.read_csv(telemetry_path)
    forbidden_cols = sorted(FORBIDDEN_INFERENCE_COLUMNS & set(frame.columns))
    if forbidden_cols:
        raise ValueError(f"labels are forbidden in inference telemetry: {forbidden_cols}")
    if "telemetry_name" in frame.columns:
        forbidden_values = sorted(
            FORBIDDEN_TELEMETRY_NAMES & set(frame["telemetry_name"].astype(str)))
        if forbidden_values:
            raise ValueError(
                "labels are forbidden in inference telemetry (telemetry_name): "
                f"{forbidden_values}")

    text = Path(metadata_path).read_text(encoding="utf-8")
    metadata = yaml.safe_load(text) if str(metadata_path).endswith((".yaml", ".yml")) \
        else json.loads(text)
    name = requested_name or metadata["prediction"]["primary_telemetry"]
    selected = frame.loc[frame["telemetry_name"] == name].copy()
    if selected.empty:
        raise ValueError(f"telemetry not found: {name}")
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], utc=True, errors="raise")
    selected = selected.sort_values(["component_id", "timestamp"], kind="stable")
    return PredictionRequest(selected, metadata, name)


def write_prediction(output_path: Path, prediction: dict) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(prediction, ensure_ascii=False, indent=2, sort_keys=True)
    (output_path / "prediction.json").write_text(payload + "\n", encoding="utf-8")
