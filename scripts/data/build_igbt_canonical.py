#!/usr/bin/env python3
"""IGBT → canonical 4 维源域构建 (§4c k-shot 源域臂实验)。

输入: igbt_features.h5 (NASA IGBT square-gate 特征, 4 器件全删失, [VCE_drift, T_case_C])
输出:
  1. igbt_canonical.h5          — 4 器件 canonical (device_canonical_v1 同 schema)
  2. mosfet_igbt_canonical.h5   — MOSFET(42) + IGBT(4) 合并多源 canonical

映射口径 (与 canonical_device.to_canonical_source 同构, 差异处显式标注):
  p_drift_norm = VCE_drift / delta_VCE   (delta_VCE=0.05, Celaya IGBT 老化文献参考值,
                                          仅作归一尺度 — IGBT 不发明 EOL 判定, 全删失口径保持)
  T_dev_C      = T_case_C
  duty         = 1.0   (占位: 加速老化=持续应力近似, IGBT 台架无 duty 记录)
  drive_norm   = 1.0   (占位: 标称驱动)
  hi           = clip(VCE_drift / delta_VCE, 0, 1)   (趋势监督; reader 侧 isotonic 去噪)
  rul_s        = NaN, event_observed=False, rul_lower_bound_s = 删失下界
  512 窗等间隔重采样 (与 MOSFET canonical 的 n_win=512 口径一致)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.preprocess.canonical_device import (       # noqa: E402
    CANONICAL_COLS, resample_to_windows)
from src.data.preprocess.nasa_mat_common import write_feature_h5   # noqa: E402

DELTA_VCE = 0.05
N_WIN = 512
SRC_DIR = ROOT / "data/features/phased_array/schema_v4/source"


def build_igbt(igbt_feat_h5: Path, out_path: Path) -> int:
    devices: dict[str, pd.DataFrame] = {}
    with h5py.File(igbt_feat_h5, "r") as f:
        for did in sorted(f["devices"].keys()):
            d = f["devices"][did]
            vce = d["x"][:, 0].astype(np.float64)
            tcase = d["x"][:, 1].astype(np.float64)
            t_s = d["elapsed_time_s"][:].astype(np.float64)
            lb = d["rul_lower_bound_s"][:].astype(np.float64)
            x4 = np.stack([vce / DELTA_VCE, tcase,
                           np.full(len(vce), 1.0), np.full(len(vce), 1.0)], axis=1)
            hi = np.clip(vce / DELTA_VCE, 0.0, 1.0)
            # 512 窗重采样: 特征+标签列一起插值 (归一寿命进度等间隔, 删失安全)
            block = np.column_stack([x4, hi, lb, t_s])
            rb = resample_to_windows(block, t_s, n_win=N_WIN)
            cf = pd.DataFrame(rb[:, :4], columns=CANONICAL_COLS)
            cf["hi"] = rb[:, 4]
            cf["rul_s"] = np.nan          # 全删失: 无精确 RUL (不发明 EOL 判定)
            cf["rul_lower_bound_s"] = rb[:, 5]
            cf["elapsed_time_s"] = rb[:, 6]
            cf["event_observed"] = False
            devices[f"igbt_{did}"] = cf
    metadata = {
        "dataset_id": "NASA_IGBT_canonical_real",
        "schema_version": "2.0",
        "canonical_schema": "device_canonical_v1",
        "feature_names": list(CANONICAL_COLS),
        "feature_dim": 4,
        "source_mode": "canonical_nasa_igbt",
        "delta_VCE": DELTA_VCE,
        "delta_note": "文献参考值仅作归一尺度, 非 EOL 判定; 全删失 (event=False, rul=NaN)",
        "placeholder_cols": {"duty": "1.0 (持续应力近似)", "drive_norm": "1.0 (标称)"},
        "n_win": N_WIN,
        "synthetic": False,
    }
    write_feature_h5(out_path, devices, feature_columns=CANONICAL_COLS, metadata=metadata)
    print(f">> igbt canonical -> {out_path.relative_to(ROOT)} ({len(devices)} devices)")
    return len(devices)


def merge_multi(mosfet_h5: Path, igbt_h5: Path, out_path: Path) -> int:
    merged: dict[str, pd.DataFrame] = {}
    meta_src = None
    for src in (mosfet_h5, igbt_h5):
        with h5py.File(src, "r") as f:
            meta_src = json.loads(f.attrs.get("metadata_json", "{}"))
            for did in sorted(f["devices"].keys()):
                d = f["devices"][did]
                frame = {c: d[c][:].astype(np.float64)
                         for c in ("x", "hi", "rul_s", "rul_lower_bound_s", "elapsed_time_s")
                         if c in d}
                cf = pd.DataFrame(frame.pop("x"), columns=CANONICAL_COLS)
                for k, v in frame.items():
                    cf[k] = v
                cf["event_observed"] = bool(d.attrs["event_observed"])
                merged[did] = cf
    metadata = {
        "dataset_id": "NASA_MOSFET+IGBT_multisource_canonical",
        "schema_version": "2.0",
        "canonical_schema": "device_canonical_v1",
        "feature_names": list(CANONICAL_COLS),
        "feature_dim": 4,
        "source_mode": "canonical_nasa_multisource",
        "components": ["mosfet_canonical (42 dev)", "igbt_canonical (4 dev, 全删失)"],
        "synthetic": False,
    }
    write_feature_h5(out_path, merged, feature_columns=CANONICAL_COLS, metadata=metadata)
    print(f">> multisource canonical -> {out_path.relative_to(ROOT)} ({len(merged)} devices)")
    return len(merged)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--igbt-features", default=str(SRC_DIR / "igbt_features.h5"))
    ap.add_argument("--mosfet-canonical", default=str(SRC_DIR / "mosfet_canonical.h5"))
    args = ap.parse_args()
    n_igbt = build_igbt(Path(args.igbt_features), SRC_DIR / "igbt_canonical.h5")
    n_all = merge_multi(Path(args.mosfet_canonical),
                        SRC_DIR / "igbt_canonical.h5",
                        SRC_DIR / "mosfet_igbt_canonical.h5")
    print(f">> 完成: IGBT {n_igbt} + 合并 {n_all} 器件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
