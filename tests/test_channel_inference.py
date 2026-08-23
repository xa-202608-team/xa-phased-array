# -*- coding: utf-8 -*-
"""F1 批2: 通道级推理数据链 (channel_inference) 契约测试。

覆盖三特性 (收口清单原文):
  - 无标签: h5 只有 x_ch (无 hi_ch/rul_*) 也能加载 — 推理遥测不依赖监督信号
  - 窗口末端: 每样本携带 window_end (=窗口末点在通道内的绝对下标), 与训练窗口语义一致
  - 特征校验: canonical 4 维宽度 / 有限性 (NaN/Inf 拒绝) / 期望维数一致
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from src.transfer.channel_inference import (
    ChannelInferenceDataset, load_channel_inference, normalize_with_stats)

H_FIXTURE = 11688.0


def _write_inference_h5(path, with_labels: bool = False, T: int = 12,
                        width: int = 4, nan_at=None):
    """写最小 v2 channel_features.h5。with_labels=False 时子组**只有 x_ch** —
    证明推理加载器零标签依赖 (契约 §9 同源纪律)。"""
    with h5py.File(path, "w") as f:
        f.attrs["channel_label_schema"] = "channel_label_v2"
        f.attrs["rul_scale_windows"] = H_FIXTURE
        f.attrs["sample_period_s"] = 21600.0
        f.attrs["mission_horizon_windows"] = H_FIXTURE
        f.attrs["rul_capped"] = "false"
        f.attrs["t_dev_unit"] = "degC"
        for ti in range(2):
            traj = f.create_group(f"traj_{ti:03d}")
            for si in range(2):
                sub = traj.create_group(f"sub_{si:02d}")
                sub.attrs["sub_id"] = si
                base = ti * 100 + si * 10
                x = (np.arange(T * width, dtype=np.float32).reshape(T, width)
                     + base)
                if nan_at is not None:
                    x[nan_at, 0] = np.nan
                sub.create_dataset("x_ch", data=x)
                if with_labels:
                    sub.create_dataset("hi_ch", data=np.linspace(0, 1, T))
                    sub.create_dataset("rul_ch_windows",
                                       data=np.arange(T, 0, -1, dtype=np.float32))
                    sub.create_dataset("rul_ch_norm",
                                       data=np.arange(T, 0, -1, dtype=np.float32)
                                       / H_FIXTURE)
    return path


# ---------------------------------------------------------------- 无标签
def test_load_label_free_h5(tmp_path):
    """h5 子组只有 x_ch (无任何标签数据集) 也能加载 — 推理零标签依赖。"""
    p = _write_inference_h5(tmp_path / "nolabel.h5", with_labels=False)
    x, ck, tid, sid, meta = load_channel_inference(p)
    assert x.shape == (2 * 2 * 12, 4)
    assert meta["channel_label_schema"] == "channel_label_v2"
    assert meta["rul_scale_windows"] == H_FIXTURE
    # channel_key 约定与训练侧一致: traj*16+sub
    assert set(np.unique(ck)) == {0 * 16 + 0, 0 * 16 + 1, 1 * 16 + 0, 1 * 16 + 1}
    assert set(np.unique(tid)) == {0, 1} and set(np.unique(sid)) == {0, 1}


def test_load_indifferent_to_labels(tmp_path):
    """同一份数据, 有无标签数据集读出的 x 完全一致 (标签不进入推理路径)。"""
    p_lab = _write_inference_h5(tmp_path / "lab.h5", with_labels=True)
    p_raw = _write_inference_h5(tmp_path / "raw.h5", with_labels=False)
    x1, *_ = load_channel_inference(p_lab)
    x2, *_ = load_channel_inference(p_raw)
    np.testing.assert_array_equal(x1, x2)


# ---------------------------------------------------------------- 特征校验
def test_load_rejects_wrong_width(tmp_path):
    p = _write_inference_h5(tmp_path / "w.h5", width=5)
    with pytest.raises(ValueError, match="特征维"):
        load_channel_inference(p)


def test_load_rejects_nan(tmp_path):
    p = _write_inference_h5(tmp_path / "nan.h5", nan_at=3)
    with pytest.raises(ValueError, match="NaN|Inf|有限"):
        load_channel_inference(p)


def test_load_drop_features(tmp_path):
    p = _write_inference_h5(tmp_path / "drop.h5")
    x, *_ = load_channel_inference(p, drop_features=["T_dev_C", "duty"])
    assert x.shape[1] == 2


def test_load_rejects_missing_x_ch(tmp_path):
    p = _write_inference_h5(tmp_path / "ok.h5")
    with h5py.File(p, "a") as f:
        del f["traj_000/sub_00/x_ch"]
    with pytest.raises((ValueError, KeyError)):
        load_channel_inference(p)


# ---------------------------------------------------------------- 窗口末端
def test_dataset_window_semantics():
    """窗口语义与训练侧 TargetSeqDataset 一致: starts=range(0,T-L+1,stride),
    window_end=start+L-1; 窗口不跨通道; 内容=原序列切片。"""
    # 2 条通道 (ck=0,1), T=10, L=4, stride=2 → starts {0,2,4,6} → 4 窗/通道
    T, L, stride = 10, 4, 2
    x = np.arange(2 * T * 4, dtype=np.float32).reshape(2 * T, 4)
    ck = np.repeat([0, 1], T)
    ds = ChannelInferenceDataset(x, ck, L=L, stride=stride)
    assert len(ds) == 8                        # 2 通道 × 4 窗
    ends_by_ch: dict[int, list[int]] = {}
    for i in range(len(ds)):
        w, k, end = ds[i]
        assert isinstance(w, torch.Tensor) and w.shape == (L, 4)
        ends_by_ch.setdefault(int(k), []).append(int(end))
        # 内容 = 该通道序列 [end-L+1 : end+1]
        chan = x[ck == int(k)]
        np.testing.assert_array_equal(w.numpy(), chan[int(end) - L + 1: int(end) + 1])
    assert ends_by_ch[0] == [3, 5, 7, 9] and ends_by_ch[1] == [3, 5, 7, 9]


def test_dataset_skips_short_channels():
    """T < L 的通道跳过并计数 (不 pad、不报错), n_skipped 如实暴露。"""
    L = 4
    x = np.arange((10 + 3) * 4, dtype=np.float32).reshape(-1, 4)
    ck = np.repeat([7, 8], [10, 3])            # ck=8 仅 3 点 < L
    ds = ChannelInferenceDataset(x, ck, L=L, stride=2)
    assert len(ds) == 4 and ds.n_skipped_channels == 1
    assert all(int(ds[i][1]) == 7 for i in range(len(ds)))


def test_dataset_feature_validation():
    x = np.zeros((8, 3), dtype=np.float32)     # 3 维
    ck = np.repeat([0], 8)
    with pytest.raises(ValueError, match="特征维"):
        ChannelInferenceDataset(x, ck, L=4, stride=1, expected_features=4)


def test_dataset_all_channels_too_short_raises():
    """全部通道都短于 L → 无样本属配置错误, 显式报错 (不静默空集)。"""
    x = np.zeros((3, 4), dtype=np.float32)
    ck = np.repeat([0], 3)
    with pytest.raises(ValueError, match="无完整窗口"):
        ChannelInferenceDataset(x, ck, L=4)


# ---------------------------------------------------------------- 归一化
def test_normalize_with_stats():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = normalize_with_stats(x, np.array([2.0, 3.0]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(out, [[-1.0, -1.0], [1.0, 1.0]], rtol=1e-6)


def test_normalize_dimension_mismatch():
    x = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="维"):
        normalize_with_stats(x, np.zeros(3), np.ones(3))
