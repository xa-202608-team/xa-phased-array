# -*- coding: utf-8 -*-
"""相控阵因果单指标预测器（method=causal_single_telemetry）。

定位（docs/MODELING.md §5）：
- **未知在轨数据接入 + 趋势基线**：只对历史观测做最小二乘线性拟合并外推，
  斜率不朝失效阈值运动时不给 RUL（返回 None），绝不使用任何未来信息。
- 与 PyTorch 主模型（src/experiments/run_groups.py，多特征 TCN/GRU + HI 监督）
  是**两套独立口径**：本入口用于单遥测接入与健康趋势研判，
  其外推误差不与主模型的 RMSE/PHM 对比实验指标混同。

checkpoint 约定（契约模板要求）：`--checkpoint` 传入时只做存在性与可加载性
校验（torch.load），加载失败以非零退出码失败，不得回退；本因果方法按定义
不依赖训练权重，数值计算不使用 checkpoint 内容。
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from component.io import PredictionRequest

SCHEMA_VERSION = "1.1.0"
CONTRACT_VERSION = "component-contract-v1.1.0"
COMPONENT = "phased_array"
METHOD = "causal_single_telemetry"

# rul_unit（元数据 prediction.rul_unit）→ 换算到秒的因子；windows 由采样步长决定
_UNIT_TO_SECONDS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0,
                    "days": 86400.0, "windows": None}


def causal_linear_forecast(values: np.ndarray, horizon: int) -> tuple[np.ndarray, float]:
    """只用历史观测 `values`（按时间升序）做最小二乘线性拟合并外推。

    返回 (horizon 步预测值, 拟合斜率/步)。不含任何平滑、季节或未来信息。
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("causal_linear_forecast 需要至少 2 个历史观测")
    if horizon < 1:
        raise ValueError("horizon 必须 >= 1")
    n = len(values)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    future_x = np.arange(n, n + horizon, dtype=float)
    forecast = intercept + slope * future_x
    return forecast, float(slope)


def estimate_rul(last_value: float, slope: float, threshold: float,
                 direction: str) -> float | None:
    """线性外推下到达失效阈值所需步数；斜率不朝阈值运动时返回 None。

    - direction=decreasing：失效 = value <= threshold，仅 slope < 0 有有限 RUL；
    - direction=increasing：失效 = value >= threshold，仅 slope > 0 有有限 RUL；
    - 已越过阈值返回 0.0（schema 允许 rul >= 0）。
    """
    if direction not in ("increasing", "decreasing"):
        raise ValueError(f"unknown degradation_direction: {direction}")
    if slope == 0.0:
        return None
    if direction == "decreasing":
        if last_value <= threshold:
            return 0.0
        return (last_value - threshold) / (-slope) if slope < 0 else None
    # increasing
    if last_value >= threshold:
        return 0.0
    return (threshold - last_value) / slope if slope > 0 else None


def current_git_commit() -> str:
    """40 位 hex 提交哈希：git rev-parse 优先，XA_GIT_COMMIT 兜底，均失败即报错。"""
    env = os.environ.get("XA_GIT_COMMIT", "").strip()
    if len(env) == 40 and all(c in "0123456789abcdef" for c in env):
        return env
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL)
        commit = out.decode().strip()
        if len(commit) == 40:
            return commit
    except (OSError, subprocess.CalledProcessError):
        pass
    raise RuntimeError(
        "无法确定 git_commit（不在 git 仓库且未设置 XA_GIT_COMMIT）；"
        "导出环境请设置 XA_GIT_COMMIT=<40 位 hex>")


class CausalSingleTelemetryPredictor:
    """method=causal_single_telemetry：单指标因果线性趋势基线。"""

    method = METHOD
    component = COMPONENT

    def __init__(self, checkpoint: Path | None = None):
        self._checkpoint = Path(checkpoint) if checkpoint is not None else None
        if self._checkpoint is not None:
            if not self._checkpoint.is_file():
                raise FileNotFoundError(f"checkpoint 不存在: {self._checkpoint}")
            import torch
            torch.load(self._checkpoint, map_location="cpu", weights_only=True)

    # ------------------------------------------------------------------
    def predict(self, request: PredictionRequest) -> dict:
        meta = request.metadata["prediction"]
        horizon = int(meta["forecast_horizon"])
        direction = str(meta["degradation_direction"])
        threshold = float(meta["failure_threshold"])
        rul_unit = str(meta.get("rul_unit", "windows"))

        unit = self._telemetry_unit(request)
        forecasts: list[dict] = []
        for cid, group in request.telemetry.groupby("component_id", sort=True):
            group = group.sort_values("timestamp", kind="stable")
            values = group["value"].to_numpy(dtype=float)
            if len(values) < 2:
                continue                       # 单点无法线性拟合，跳过该个体
            forecast, slope = causal_linear_forecast(values, horizon)
            sigma = float(np.std(values - np.polyval(np.polyfit(
                np.arange(len(values), dtype=float), values, 1),
                np.arange(len(values), dtype=float)), ddof=2 if len(values) > 2 else 1))
            last_ts = group["timestamp"].iloc[-1].to_pydatetime()
            step_seconds = self._step_seconds(group["timestamp"])
            rul_steps = estimate_rul(float(values[-1]), slope, threshold, direction)
            for h in range(1, horizon + 1):
                item = {
                    "component_id": str(cid),
                    "origin_timestamp": _iso(last_ts),
                    "horizon_step": h,
                    "predicted_timestamp": _iso(last_ts + timedelta(
                        seconds=step_seconds * h)),
                    "predicted_value": float(forecast[h - 1]),
                    "unit": unit,
                    "uncertainty_lower": float(forecast[h - 1] - 1.96 * sigma),
                    "uncertainty_upper": float(forecast[h - 1] + 1.96 * sigma),
                }
                if rul_steps is not None:
                    item["rul"] = self._rul_in_unit(rul_steps, step_seconds, rul_unit)
                    item["rul_unit"] = rul_unit
                forecasts.append(item)
        if not forecasts:
            raise ValueError(f"没有可预测个体 (telemetry_name={request.telemetry_name})")
        return {
            "schema_version": SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "component": COMPONENT,
            "git_commit": current_git_commit(),
            "telemetry_name": request.telemetry_name,
            "generated_at": _iso(datetime.now(timezone.utc)),
            "forecasts": forecasts,
            "status": "PREDICTION_OK",
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _telemetry_unit(request: PredictionRequest) -> str:
        for channel in request.metadata.get("telemetry", []):
            if channel.get("name") == request.telemetry_name:
                return str(channel["unit"])
        if "unit" in request.telemetry.columns:
            modes = request.telemetry["unit"].dropna().astype(str)
            if not modes.empty:
                return str(modes.mode().iloc[0])
        raise ValueError(f"无法确定 {request.telemetry_name} 的单位")

    @staticmethod
    def _step_seconds(timestamps: pd.Series) -> float:
        diffs = timestamps.diff().dropna().dt.total_seconds()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            raise ValueError("时间戳无法推出采样步长")
        return float(diffs.median())

    @staticmethod
    def _rul_in_unit(rul_steps: float, step_seconds: float, rul_unit: str) -> float:
        factor = _UNIT_TO_SECONDS.get(rul_unit)
        if factor is None:                     # windows：直接以窗数计
            return round(float(rul_steps), 3)
        return round(float(rul_steps) * step_seconds / factor, 3)


def build_predictor(checkpoint: Path | None = None) -> CausalSingleTelemetryPredictor:
    """契约入口：返回提供 predict(request)->dict 的预测器。"""
    return CausalSingleTelemetryPredictor(checkpoint)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
