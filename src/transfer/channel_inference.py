"""transfer/channel_inference.py — 通道级推理数据链 (F1 批2 收口)。

与 channel_dataset.py (训练侧) 的三点区别 (收口清单原文):
  - **无标签**: 只打开 x_ch 数据集与 sub_id attr, 绝不触碰 hi_ch / rul_ch_windows /
    rul_ch_norm — 推理遥测零监督信号依赖 (契约 §9 标签隔离的同源纪律)。
  - **窗口末端**: 每样本携带 window_end (= 窗口末点在通道序列内的绝对下标),
    预测结果由此对齐回通道时间轴; 窗口切法与训练侧 TargetSeqDataset 逐字一致
    (starts = range(0, T-L+1, stride)), 不跨通道、短通道跳过不 pad。
  - **特征校验**: canonical 4 维宽度 + 有限性 (NaN/Inf 拒绝) + 与 bundle 期望维数一致,
    维数/内容不符即 fail (不 warning 继续)。

物理量还原 (F1-A §9 例外口径): rul_norm → rul_windows (×rul_scale_windows=H) →
rul_days (×sample_period_s/86400); 相对寿命比例仍须除该通道自身真实 EOL。
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.sim.build_channel_hi import CANONICAL_COLS
from src.transfer.channel_dataset import read_channel_label_meta


def load_channel_inference(h5_path: Path | str, drop_features: list | None = None,
                           n_sub_per_traj: int = 16):
    """读 channel_features.h5 → 通道级扁平数组 (**仅特征, 无任何标签**)。

    返回 (x_ch (N,F), channel_keys (N,), traj_ids (N,), sub_ids (N,), meta dict)。
    meta = read_channel_label_meta 产物 (H/sample_period_s 等, 推理侧物理还原用)。

    特征校验: x_ch 必须为 canonical 宽度 len(CANONICAL_COLS)=4 且全部有限;
    drop_features 按 CANONICAL_COLS 名删列 (与训练侧同语义)。
    子组缺 x_ch / sub_id → ValueError (数据损坏显式失败, 不静默跳过)。
    """
    drop_features = drop_features or []
    unknown = [c for c in drop_features if c not in CANONICAL_COLS]
    if unknown:
        raise ValueError(f"drop_features 含未知列 {unknown}; 合法={CANONICAL_COLS}")
    x_list, ck_list, tid_list, sid_list = [], [], [], []
    with h5py.File(h5_path, "r") as f:
        meta = read_channel_label_meta(f)          # v1/v2 强校验复用 (尺度元数据非标签)
        for tk in sorted(f.keys()):
            traj_grp = f[tk]
            traj_id = int(tk.split("_")[1])
            for sk in sorted(k for k in traj_grp.keys() if k.startswith("sub_")):
                sub = traj_grp[sk]
                if "sub_id" not in sub.attrs:
                    raise ValueError(f"{tk}/{sk} 缺 sub_id attr (数据损坏)")
                if "x_ch" not in sub:
                    raise ValueError(f"{tk}/{sk} 缺 x_ch 数据集 (数据损坏)")
                sub_id = int(sub.attrs["sub_id"])
                x_ch = sub["x_ch"][:].astype(np.float32)
                if x_ch.ndim != 2 or x_ch.shape[1] != len(CANONICAL_COLS):
                    raise ValueError(
                        f"{tk}/{sk} x_ch 特征维 {x_ch.shape} != canonical "
                        f"{len(CANONICAL_COLS)} 维 ({CANONICAL_COLS})")
                if not np.isfinite(x_ch).all():
                    raise ValueError(f"{tk}/{sk} x_ch 含 NaN/Inf (非有限值拒绝)")
                if drop_features:
                    keep = [i for i, c in enumerate(CANONICAL_COLS)
                            if c not in drop_features]
                    x_ch = x_ch[:, keep]
                x_list.append(x_ch)
                ck_list.append(np.full(x_ch.shape[0], traj_id * n_sub_per_traj + sub_id))
                tid_list.append(np.full(x_ch.shape[0], traj_id))
                sid_list.append(np.full(x_ch.shape[0], sub_id))
    if not x_list:
        raise RuntimeError(f"channel_features.h5 无子阵数据: {h5_path}")
    return (np.concatenate(x_list), np.concatenate(ck_list),
            np.concatenate(tid_list), np.concatenate(sid_list), meta)


class ChannelInferenceDataset(Dataset):
    """推理窗口数据集: (x_window (L,F), channel_key, window_end) 三元组。

    窗口语义与训练侧 TargetSeqDataset 逐字一致: 每通道 starts=range(0, T-L+1, stride),
    window = x[start:start+L], window_end = start+L-1 (通道内绝对下标);
    窗口绝不跨通道; T < L 的通道跳过并计入 n_skipped_channels。
    归一化不在本类做 — 调用方用 normalize_with_stats(bundle 统计量) 先归一。
    """

    def __init__(self, x: np.ndarray, channel_keys: np.ndarray, L: int,
                 stride: int = 1, expected_features: int | None = None):
        if x.ndim != 2:
            raise ValueError(f"x须为 (N,F) 二维, 得 {x.shape}")
        if expected_features is not None and x.shape[1] != expected_features:
            raise ValueError(
                f"特征维不匹配: 数据 {x.shape[1]} != 期望 {expected_features} "
                "(bundle 与遥测不一致, 拒绝推理)")
        if len(x) != len(channel_keys):
            raise ValueError(f"x({len(x)}) 与 channel_keys({len(channel_keys)}) 行数不一致")
        if L < 1 or stride < 1:
            raise ValueError(f"L/stride 须为正, 得 L={L} stride={stride}")
        self.L, self.stride = int(L), int(stride)
        self.n_features = int(x.shape[1])
        self.n_skipped_channels = 0
        self.samples: list[tuple[np.ndarray, int, int]] = []   # (window, ck, end)
        df = pd.DataFrame({"ck": channel_keys})
        for _, g in df.groupby("ck"):
            idxs = g.index.to_numpy()
            T = len(idxs)
            if T < L:
                self.n_skipped_channels += 1
                continue
            for s in range(0, T - L + 1, self.stride):
                self.samples.append((
                    x[idxs[s:s + L]].astype(np.float32),
                    int(channel_keys[idxs[s]]),
                    int(s + L - 1),
                ))
        if not self.samples:
            raise ValueError(
                f"无完整窗口 (全部 {self.n_skipped_channels} 条通道均短于 L={L}); "
                "检查输入序列长度或减小 L")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        w, ck, end = self.samples[i]
        return (torch.from_numpy(w), torch.tensor(ck, dtype=torch.long),
                torch.tensor(end, dtype=torch.long))


def eligible_channel_keys(channel_keys: np.ndarray, L: int) -> np.ndarray:
    """按升序返回窗口数达标 (T >= L) 的通道键。

    --limit-channels 场景必须从合格通道中按序截取; 否则在含早失效短通道的数据上
    可能取到全池 T < L, 使 ChannelInferenceDataset 无完整窗口可用 (F6 回归)。
    """
    keys, counts = np.unique(channel_keys, return_counts=True)
    return keys[counts >= int(L)]


def normalize_with_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """按训练侧统计量 z-score 归一 (与 run_groups 训练路径同式: (x-mean)/std)。

    mean/std 维数必须与 x 特征维一致, 否则 ValueError (bundle 与数据不匹配)。
    """
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (x.shape[1],) or std.shape != (x.shape[1],):
        raise ValueError(
            f"归一统计量维 {mean.shape}/{std.shape} 与特征维 {x.shape[1]} 不一致")
    return ((x - mean) / std).astype(np.float32)
