# -*- coding: utf-8 -*-
"""相控阵组件统一单指标遥测预测入口（component-contract v1.1）。

`python -m component.predict` 提供离线单次预测：
输入长表遥测 CSV（schemas/telemetry.schema.json）+ 数据集元数据
（schemas/dataset-metadata.schema.json），输出预测文档
（schemas/prediction.schema.json，component=phased_array）。

方法定位（docs/MODELING.md §5）：本入口为**未知在轨数据接入 + 因果趋势基线**
（method=causal_single_telemetry），只做历史观测线性外推；
不与 PyTorch 主模型（src/experiments/run_groups.py）的 RMSE/PHM 指标口径混同。

`python -m component.predict_gru`（F1 批3）：通道级 GRU RUL 推理入口
（method=gru_channel_rul），消费 run_groups --export-inference-dir 导出的
val-best bundle，对 channel_features.h5 x_ch 特征（无标签读）输出
rul_prediction.json（schemas/rul-prediction.schema.json）。
"""
from component.predictor import CausalSingleTelemetryPredictor, build_predictor

METHOD = "causal_single_telemetry"
COMPONENT = "phased_array"

__all__ = ["METHOD", "COMPONENT", "CausalSingleTelemetryPredictor", "build_predictor"]
