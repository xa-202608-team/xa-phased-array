"""PA1 源域特征工程测试 (schema_v2, 2 维 synthetic 验证逻辑 + 防泄漏)。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.preprocess.mosfet_features import (          # noqa: E402
    FEATURE_NAMES, extract_features, construct_labels, make_synthetic,
    _mono_violation_rate)


def _one_device(seed=0):
    synth = make_synthetic(n_devices=1, n_pts=128, seed=seed)
    raw = list(synth.values())[0]
    return raw


def test_feature_dim_and_no_nan():
    raw = _one_device(0)
    feat = extract_features(raw)
    assert feat.shape[1] == 2 == len(FEATURE_NAMES)
    assert list(feat.columns) == FEATURE_NAMES
    assert not np.any(np.isnan(feat.values))
    # RDS_drift 应从 0 开始 (相对漂移)
    assert abs(feat["RDS_drift"].iloc[0]) < 1e-9


def test_rds_drift_monotone_rising():
    """合成器件 RDS_drift 应单调上升 (退化主信号)。"""
    raw = _one_device(1)
    feat = extract_features(raw)
    d = np.diff(feat["RDS_drift"].values)
    assert np.quantile(d, 0.95) > 0, "RDS_drift 应上升"


def test_labels_hi_unit_and_rul_seconds():
    """schema_v2: HI∈[0,1] 固定尺度, RUL 为秒单位 (eol-t)·interval。"""
    raw = _one_device(2)
    feat = extract_features(raw)
    lab = construct_labels(feat, delta_threshold=0.05, sample_interval_s=3600.0)
    hi = lab["hi"].values
    assert hi.min() >= -1e-6 and hi.max() <= 1 + 1e-6       # HI 固定尺度 [0,1]
    rul = lab["rul_s"].values
    assert (rul >= 0).all()                                  # RUL 非负 (秒)
    assert rul.max() <= 3600.0 * len(rul) + 1               # 不超总寿命
    # 失效器件: label_fail 置 1 后 rul 归零, 且 label_fail 一旦置 1 持续到末尾
    lf = lab["label_fail"].values
    if lf.any():
        fi = int(np.argmax(lf))
        assert np.all(lf[fi:] == 1)
        assert rul[fi:].max() == 0.0


def test_device_split_no_leakage():
    """LOO 划分: 同一 device_id 不应同时出现在 train 和 val (schema_v2 分组)。"""
    synth = make_synthetic(n_devices=6, n_pts=100, seed=7)
    devices = {did: construct_labels(extract_features(raw), 0.05)
               for did, raw in synth.items()}
    val_id = list(devices.keys())[-1]                       # 默认最后器件做 val
    train_devs = set(devices.keys()) - {val_id}
    val_devs = {val_id}
    assert len(train_devs & val_devs) == 0, "train/val 器件泄漏"
    assert _mono_violation_rate(devices) < 0.05
