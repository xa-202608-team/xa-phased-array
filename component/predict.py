# -*- coding: utf-8 -*-
"""`python -m component.predict` 命令行入口（契约 §2 / §9.3）。

只做参数解析与调用；算法在 component/predictor.py，IO 隔离在 component/io.py。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from component.io import load_request, write_prediction
from component.predictor import build_predictor


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m component.predict",
        description="相控阵单指标遥测因果趋势预测（method=causal_single_telemetry）")
    parser.add_argument("--telemetry", type=Path, required=True,
                        help="长表遥测 CSV（schemas/telemetry.schema.json）")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="数据集元数据 YAML/JSON（schemas/dataset-metadata.schema.json）")
    parser.add_argument("--output", type=Path, required=True,
                        help="输出目录，写入 prediction.json")
    parser.add_argument("--telemetry-name", default=None,
                        help="主遥测通道名；缺省取 metadata prediction.primary_telemetry")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="可选；传入时仅做存在性/可加载性校验（失败即非零退出），"
                             "因果方法数值计算不使用权重")
    args = parser.parse_args()

    request = load_request(args.telemetry, args.metadata, args.telemetry_name)
    prediction = build_predictor(args.checkpoint).predict(request)
    write_prediction(args.output, prediction)
    print(f">> prediction.json -> {args.output} "
          f"({len(prediction['forecasts'])} forecasts, "
          f"telemetry={prediction['telemetry_name']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
