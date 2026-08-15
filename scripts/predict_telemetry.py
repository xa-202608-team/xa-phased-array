# -*- coding: utf-8 -*-
"""单指标遥测预测统一入口（薄包装，透传参数与退出码）。

等价命令：
  python scripts/predict_telemetry.py --telemetry X.csv --metadata Y.yaml \
      --telemetry-name array_gain_db --output OUT

底层调 `python -m component.predict`（method=causal_single_telemetry，
输出 prediction.json 符合 schemas/prediction.schema.json）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="相控阵单指标遥测因果趋势预测")
    ap.add_argument("--telemetry", required=True, help="长表遥测 CSV (telemetry.schema.json)")
    ap.add_argument("--metadata", required=True, help="数据集元数据 YAML/JSON")
    ap.add_argument("--output", required=True, help="输出目录 (写入 prediction.json)")
    ap.add_argument("--telemetry-name", default=None, help="主遥测通道；缺省取元数据 primary_telemetry")
    ap.add_argument("--checkpoint", default=None, help="可选；传入时仅做加载校验")
    args = ap.parse_args()

    cmd = [sys.executable, "-m", "component.predict",
           "--telemetry", args.telemetry,
           "--metadata", args.metadata,
           "--output", args.output]
    if args.telemetry_name:
        cmd += ["--telemetry-name", args.telemetry_name]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
