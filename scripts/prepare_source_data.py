# -*- coding: utf-8 -*-
"""源域数据准备统一入口（judge/full 复现的第一步）。

三级策略（全部落 `pretrain.canonical_source_path`，下游统一走 canonical 4 维）：
1. canonical H5 已存在（真实 NASA MOSFET 转换产物）→ 直接使用，source_mode=canonical_nasa；
2. schema_v3 真实特征存在 → 转 canonical（to_canonical_source），source_mode=nasa_real；
3. 都不存在 → `mosfet_features --synthetic` 生成合成源域，再转 canonical，
   **source_mode=synthetic**（结果不得与正式源域混用，manifest 必须标记）。

真实 NASA 数据（7.85GB MAT）不随仓库分发、不自动下载；需要正式源域时按
docs/SIMULATION_REPRODUCE.md 的手动步骤下载并运行 mosfet_real_loader_v2。

用法：
  python scripts/prepare_source_data.py [--config configs/phased_array.yaml]
                                        [--output outputs/source_report.json]
输出：canonical H5（就绪时）+ JSON 报告（source_mode / sha256 / 路径相对仓库根）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess.canonical_device import CANONICAL_COLS, V3_COLS, to_canonical_source  # noqa: E402
from src.data.preprocess.mosfet_features import (                                    # noqa: E402
    construct_labels, extract_features, make_synthetic)
from src.data.preprocess.nasa_mat_common import write_feature_h5                     # noqa: E402
from src.utils import load_config                                                    # noqa: E402


def _synthetic_v3_devices(seed: int, delta_thr: float) -> dict[str, pd.DataFrame]:
    """合成 schema_v3 5 维源域（绕开 mosfet_features CLI 的 target_dim 断言遗留）。

    前两维复用 mosfet_features.make_synthetic + extract_features（RDS_drift/T_case_C），
    后三维为合成工况常量（supply_V/gate_voltage/duty_cycle）——synthetic 源域仅用于
    管线连通性 smoke，不承载正式迁移结论（见 docs/MODELING.md）。
    """
    rng = np.random.default_rng(seed + 1)
    devices: dict[str, pd.DataFrame] = {}
    for did, raw in make_synthetic(seed=seed).items():
        feat = extract_features(raw)
        feat["supply_V"] = np.full(len(feat), 28.0)                 # 母线电压 (合成常量)
        feat["gate_voltage"] = np.full(len(feat), 5.0)              # 栅压 (合成常量)
        feat["duty_cycle"] = np.full(len(feat), float(rng.uniform(0.3, 0.8)))
        devices[did] = construct_labels(feat, delta_thr)
    return devices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_v2_devices(h5_path: Path) -> dict[str, pd.DataFrame]:
    devices: dict[str, pd.DataFrame] = {}
    with h5py.File(h5_path, "r") as f:
        feature_names = [s.decode() if isinstance(s, bytes) else str(s)
                         for s in f.attrs.get("feature_names", [])]
        root = f["devices"]
        for did in sorted(root.keys()):
            g = root[did]
            frame = pd.DataFrame(np.asarray(g["x"][:], dtype=np.float32),
                                 columns=feature_names or [f"f{i}" for i in range(g["x"].shape[1])])
            for col in ("hi", "rul_s", "rul_lower_bound_s", "elapsed_time_s"):
                if col in g:
                    frame[col] = np.asarray(g[col][:], dtype=np.float64)
            frame["event_observed"] = bool(g.attrs.get("event_observed", False))
            devices[str(did)] = frame
    return devices


def _write_canonical(devices: dict[str, pd.DataFrame], feature_names: list[str],
                     out_path: Path, source_mode: str, delta_r_src: float) -> None:
    canonical_devices: dict[str, pd.DataFrame] = {}
    for did, frame in devices.items():
        x4 = to_canonical_source(frame[feature_names].to_numpy(dtype=np.float64),
                                 delta_R_src=delta_r_src, feature_names=feature_names)
        cf = pd.DataFrame(x4, columns=CANONICAL_COLS)
        for col in ("hi", "rul_s", "rul_lower_bound_s", "elapsed_time_s"):
            if col in frame:
                cf[col] = frame[col].to_numpy(dtype=np.float64)
        cf["event_observed"] = bool(frame["event_observed"].iloc[0])
        canonical_devices[did] = cf
    metadata = {
        "dataset_id": f"NASA_MOSFET_canonical_{'synthetic' if source_mode == 'synthetic' else 'real'}",
        "schema_version": "2.0",
        "canonical_schema": "device_canonical_v1",
        "feature_names": list(CANONICAL_COLS),
        "feature_dim": len(CANONICAL_COLS),
        "source_mode": source_mode,
        "delta_R_src": delta_r_src,
        "synthetic": source_mode == "synthetic",
    }
    write_feature_h5(out_path, canonical_devices, feature_columns=CANONICAL_COLS,
                     metadata=metadata)
    print(f">> canonical ({source_mode}) -> {out_path.relative_to(ROOT)} "
          f"({len(canonical_devices)} devices, 4 维 {CANONICAL_COLS})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phased_array.yaml"))
    ap.add_argument("--output", default=None, help="JSON 报告输出路径（默认不写）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    canonical_path = ROOT / cfg["pretrain"]["canonical_source_path"]
    v3_path = ROOT / cfg["pretrain"]["source_feature_path"]
    delta_r_src = float(cfg["source"]["failure"]["RDS_delta_threshold"])

    report: dict = {"config": str(Path(args.config).resolve().relative_to(ROOT)),
                    "canonical_path": str(canonical_path.relative_to(ROOT))}

    if canonical_path.is_file():
        # 如实标注：canonical H5 的 metadata 记录了它自身的来源
        # （synthetic 产物不得伪装成 canonical_nasa）
        with h5py.File(canonical_path, "r") as f:
            meta = json.loads(f.attrs.get("metadata_json", "{}"))
        source_mode = str(meta.get("source_mode", "canonical_nasa"))
        print(f">> canonical H5 已存在，直接使用: {canonical_path.relative_to(ROOT)} "
              f"(metadata source_mode={source_mode})")
    elif v3_path.is_file():
        source_mode = "nasa_real"
        print(f">> 真实 schema_v3 特征存在: {v3_path.relative_to(ROOT)} -> 转 canonical")
        devices = _read_v2_devices(v3_path)
        with h5py.File(v3_path, "r") as f:
            names = [s.decode() if isinstance(s, bytes) else str(s)
                     for s in f.attrs.get("feature_names", [])]
        _write_canonical(devices, names, canonical_path, source_mode, delta_r_src)
    else:
        source_mode = "synthetic"
        print(">> 无真实源域特征 -> 生成 synthetic 源域（结果不得与正式源域混用）")
        devices = _synthetic_v3_devices(cfg.get("seed", 42), delta_r_src)
        _write_canonical(devices, list(V3_COLS), canonical_path, source_mode, delta_r_src)

    report.update(source_mode=source_mode,
                  canonical_sha256=_sha256(canonical_path),
                  canonical_size_bytes=canonical_path.stat().st_size,
                  n_devices=len(_read_v2_devices(canonical_path)))
    print(f">> source_mode = {source_mode}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f">> 报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
