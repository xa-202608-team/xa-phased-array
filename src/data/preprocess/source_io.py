"""src/data/preprocess/source_io.py

统一源域特征 h5 读取: schema_v1 扁平 + schema_v2 分组 + schema_v3 工况条件化 三兼容。

schema_v1 (飞轮 wheel_features.py / 旧 mosfet synthetic):
  扁平拼接 — features(N,F) / hi(N,) / rul(N,) / t_idx(N,) / {id_field}(N,) / split(N,)

schema_v2 (NASA 真实 mosfet_real_loader / igbt_features / 新 synthetic):
  分组 — devices/{id}/x(T,F) / hi / rul_s / elapsed_time_s / .attrs(event_observed, ...)

schema_v3 (Phase 1 修复版, mosfet_real_loader_v2):
  分组 + 5 维特征 — devices/{id}/x(T,5) / hi / rul_s / elapsed_time_s / supply_V / gate_voltage / duty_cycle
  工况条件化版本，删除全局 log-MAD 过滤，保留所有工况段数据

load_source_features() 自动检测 schema, 返回统一的 SourceData 扁平视图,
供 pretrain / transfer / experiments 复用。LOO 划分严格按器件个体, 禁止
时间窗打散 (防泄漏, CLAUDE.md 开发约束 7)。

RUL 单位差异 (v1=窗索引 / v2/v3=秒) 通过 SourceData.rul_max 暴露给调用方归一,
不在 loader 内强行归一, 保持职责单一。

删失器件标签修复 (2026-07-22 D1/D6):
  - event_observed (逐行 bool): v2/v3 读 devices/{id}/.attrs['event_observed']
    按器件广播到所有行; v1 默认 True (飞轮均为真实失效轨迹)。
  - rul_lower_bound (逐行 float): v2/v3 读 rul_lower_bound_s; v1 = rul。
    pretrain 区分失效/删失: 失效行 Huber(rul_pred, rul_true),
    删失行 hinge = max(0, rul_lower_bound - rul_pred)²。
    `rul` 字段保持原语义 (失效=exact, 删失=lower_bound 兼容 NaN 回退路径),
    但 pretrain 用 event_observed 分流, 不再把 lower_bound 当精确 RUL。
  - hi 按器件做 isotonic 去噪 (v2/v3 only): 源 HI=clip(raw_drift/threshold) 含噪
    非单调 (mono_viol≈0.15), 目标 HI=isotonic(1-EIRP_norm) 单调, 口径不一致。
    与目标 HI 对齐, 在 loader 内完成, 不 rebuild h5。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def _decode(arr) -> list[str]:
    return [s.decode() if isinstance(s, bytes) else str(s) for s in arr]


@dataclass
class SourceData:
    """源域特征统一扁平视图 (两种 schema 归一后的内存表示)。

    features / hi / rul 沿行对齐; device_ids[i] 标记该行所属器件;
    t_index[i] 为该器件内按时间排序的序号 (供滑窗 groupby); split[i] ∈ {'train','val'}。

    event_observed[i] (逐行 bool): 该行所属器件是否观测到失效 (EOL)。
      v1 全部 True (飞轮 wheel 为真实失效轨迹);
      v2 来自 devices/{id}/.attrs['event_observed'] 按器件广播。
    rul_lower_bound[i] (逐行 float): RUL 的保守下界 (censor_time - t)。
      失效器件 = exact RUL (与 rul 一致); 删失器件 = censor_time - t。
      pretrain 用此字段对删失行施 hinge 约束 (允许预测 > lower_bound)。
    """
    features: np.ndarray          # (N, F) float32
    hi: np.ndarray                # (N,) float32   原始单位 (固定尺度 [0,1])
    rul: np.ndarray               # (N,) float32   原始单位 (v1=窗索引 / v2=秒)
    device_ids: list              # (N,) str
    t_index: np.ndarray           # (N,) int64
    split: list                   # (N,) str       'train' / 'val'
    feature_names: list           # list[str]
    schema_version: str           # '1' / '2'
    n_devices: int
    rul_max: float                # 全局 RUL 上界 (调用方归一用, max(.,1.0))
    event_observed: np.ndarray    # (N,) bool      逐行 event 标记 (v2 attrs 广播; v1 全 True)
    rul_lower_bound: np.ndarray   # (N,) float32   逐行 RUL 下界 (失效=exact; 删失=censor-t)

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def device_id_array(self) -> np.ndarray:
        return np.array(self.device_ids, dtype=object)

    @property
    def split_array(self) -> np.ndarray:
        return np.array(self.split, dtype=object)


def _detect_schema(h5: h5py.File) -> str:
    # v4 canonical (device_canonical_v1, T3/M3): schema attrs 含 canonical
    schema_attr = str(h5.attrs.get("schema", ""))
    if "canonical" in schema_attr.lower():
        return "4"
    if "devices" in h5 and isinstance(h5["devices"], h5py.Group):
        # 检测 schema v2 vs v3（通过 feature_dim 判断）
        if "feature_dim" in h5.attrs and h5.attrs["feature_dim"] == 5:
            return "3"  # 5 维特征，工况条件化版本
        return "2"  # 2 维特征
    if "features" in h5 and isinstance(h5["features"], h5py.Dataset):
        return "1"
    raise ValueError(
        f"无法识别 source h5 schema: top-level keys={list(h5.keys())} "
        "(期望 v1 'features' 扁平拼接 或 v2/v3 'devices' 分组)"
    )


def _feature_names_from_attrs(attrs) -> list[str]:
    if "feature_names" in attrs:
        return _decode(list(attrs["feature_names"]))
    return []


def _read_v1(h5, id_field):
    features = h5["features"][:].astype(np.float32)
    hi = h5["hi"][:].astype(np.float32)
    rul = h5["rul"][:].astype(np.float32)
    t_idx = h5["t_idx"][:].astype(np.int64)
    bid = _decode(h5[id_field][:])
    if "split" in h5:
        split = _decode(h5["split"][:])
    else:
        split = ["train"] * len(bid)
    # v1 (飞轮 wheel) 全部为真实失效轨迹 → event_observed=True, rul_lower_bound=rul
    event_observed = np.ones(len(bid), dtype=bool)
    rul_lower_bound = rul.copy()
    return features, hi, rul, t_idx, bid, split, event_observed, rul_lower_bound


def _isotonic_denoise_hi(hi: np.ndarray, t: np.ndarray) -> np.ndarray:
    """对单个器件的 hi 按时间做 isotonic 去噪 (increasing)。

    源 HI=clip(raw_drift/threshold) 含测量噪声可非单调; 目标 HI=isotonic(1-EIRP_norm)
    单调上升。在 loader 内对齐口径, 避免 mono_viol≈0.5 (诊断 D6)。
    短序列/已单调序列不影响。返回与输入同形状 float32。
    """
    finite = np.isfinite(t) & np.isfinite(hi)
    if int(finite.sum()) < 10:
        return hi.astype(np.float32, copy=True)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    out = hi.astype(np.float64, copy=True)
    out[finite] = iso.fit_transform(t[finite], hi[finite].astype(np.float64))
    # 保持 [0,1] 范围 (IsotonicRegression out_of_bounds='clip' 已保单调, 但数值漂移可能微越界)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def _read_v2(h5, hi_isotonic: bool = True):
    """读取 v2 分组 h5。

    hi_isotonic: True=按器件对 hi 做 isotonic 去噪 (与目标域 HI 口径一致);
                 False=保留原始 hi (调试或对比用)。
    """
    grp = h5["devices"]
    keys = sorted(grp.keys())
    feats_list, hi_list, rul_list, bid_list, tidx_list = [], [], [], [], []
    ev_list, lb_list = [], []
    for did in keys:
        g = grp[did]
        x = g["x"][:].astype(np.float32)
        hi = g["hi"][:].astype(np.float32)
        # v2 RUL 字段名 rul_s (秒); 兼容少数 rul 命名
        if "rul_s" in g:
            rul = g["rul_s"][:].astype(np.float32)
        elif "rul" in g:
            rul = g["rul"][:].astype(np.float32)
        else:
            rul = np.full(len(x), np.nan, dtype=np.float32)
        # event_observed (器件级 attrs 广播到所有行)
        ev_device = bool(g.attrs.get("event_observed", False))
        # rul_lower_bound_s (v2 必有; 若缺则用 rul 兜底)
        if "rul_lower_bound_s" in g:
            lb = g["rul_lower_bound_s"][:].astype(np.float32)
        else:
            lb = rul.copy()
        # 删失器件 rul_s=NaN (build_fixed_threshold_labels: exact_rul 仅失效器件赋值,
        # 删失器件 exact_rul 保持 NaN) -> 用 rul_lower_bound_s 填充 rul 字段保证无 NaN,
        # 但 rul_lower_bound 字段独立保留; pretrain 用 event_observed 区分精确/下界
        nan_mask = np.isnan(rul)
        if nan_mask.any():
            rul = np.where(nan_mask, lb, rul).astype(np.float32)
        # 按时间排序 (elapsed_time_s 优先, 退化到原始顺序) — 保证滑窗时序正确
        if "elapsed_time_s" in g:
            t_raw = g["elapsed_time_s"][:].astype(np.float64)
            order = np.argsort(t_raw, kind="stable")
        else:
            t_raw = np.arange(len(x), dtype=np.float64)
            order = np.arange(len(x))
        x, hi, rul, lb = x[order], hi[order], rul[order], lb[order]
        t_sorted = t_raw[order]
        if hi_isotonic:
            hi = _isotonic_denoise_hi(hi, t_sorted)
        t_idx = np.arange(len(x), dtype=np.int64)
        ev_row = np.full(len(x), ev_device, dtype=bool)
        feats_list.append(x)
        hi_list.append(hi)
        rul_list.append(rul)
        lb_list.append(lb)
        ev_list.append(ev_row)
        bid_list.extend([did] * len(x))
        tidx_list.append(t_idx)
    features = np.concatenate(feats_list, axis=0)
    hi = np.concatenate(hi_list)
    rul = np.concatenate(rul_list)
    lb = np.concatenate(lb_list)
    t_idx = np.concatenate(tidx_list)
    event_observed = np.concatenate(ev_list)
    return features, hi, rul, t_idx, bid_list, event_observed, lb


def load_source_features(
    h5_path: str | Path,
    id_field: str = "bearing_id",
    val_device_ids: Iterable[str] | None = None,
    hi_isotonic: bool = True,
) -> SourceData:
    """读取源域特征 h5 (v1 扁平 / v2 分组自动适配) → 统一 SourceData。

    LOO 划分:
      - v1: 沿用 h5 内 split 字段 (wheel_features.py 已写入)
      - v2: h5 无 split, 由 val_device_ids 决定; 缺省 = 最后一个器件做 val
        (与旧 synthetic 单折行为一致; 正式实验应遍历所有器件做完整 LOO)

    hi_isotonic (默认 True): v2 读取时按器件对 hi 做 isotonic 去噪 (与目标 HI 口径
        一致, 修 mono_viol≈0.5 来源 D6); v1 飞轮 hi 无此问题, 参数被忽略。
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        schema = _detect_schema(f)
        feature_names = _feature_names_from_attrs(f.attrs)
        if schema == "1":
            features, hi, rul, t_idx, bid, split, event_observed, rul_lb = _read_v1(f, id_field)
        else:
            features, hi, rul, t_idx, bid, event_observed, rul_lb = _read_v2(f, hi_isotonic=hi_isotonic)
            val_ids = set(val_device_ids or [])
            if not val_ids and bid:
                val_ids = {bid[-1]}               # 缺省最后一个器件做 val
            split = ["val" if d in val_ids else "train" for d in bid]

    finite_rul = rul[np.isfinite(rul)]
    rul_max = float(np.max(finite_rul)) if finite_rul.size else 0.0
    rul_max = max(rul_max, 1.0)
    rul_lb = np.where(np.isfinite(rul_lb), rul_lb, rul).astype(np.float32)

    return SourceData(
        features=features,
        hi=hi,
        rul=rul,
        device_ids=bid,
        t_index=t_idx,
        split=split,
        feature_names=feature_names,
        schema_version=schema,
        n_devices=len(set(bid)),
        rul_max=rul_max,
        event_observed=event_observed,
        rul_lower_bound=rul_lb,
    )


# ---------------------------------------------------------------- HI 层滑窗工具
def make_hi_windows(
    hi: np.ndarray,
    device_ids,
    t_index: np.ndarray,
    L: int,
    stride: int = 1,
):
    """构造 HI 滑窗序列 + 窗末 HI 标签 (设计文档 §3.2/§4.5)。

    按 device_id groupby, 每器件内按 t_index 排序后滑窗 (窗长 L, 步长 stride)。
    严格不跨器件 (CLAUDE.md 开发约束 7: 按器件个体划分, 杜绝同一退化轨迹泄漏)。

    产物供 HISeqEncoder / HIDynamicsModel 训练或 MMD binning 用 — encoder 输入语义
    统一为 [HI, ΔHI] (跨域同形同义, 见设计文档 §2.1)。

    Args:
        hi: (N,) float32  逐行 HI 值 (源/目标通用, 已归一到 [0,1])
        device_ids: (N,) list/arr  器件/轨迹 id
        t_index: (N,) int  逐行器件内时间序号 (排序依据)
        L: int  窗长
        stride: int  步长 (默认 1)

    Returns:
        x_HI: (N', L, 2) float32
            通道 0 = HI 序列; 通道 1 = ΔHI (窗内 np.diff 前向差分, 首位补 0 保窗长 L)
            窗内首位 ΔHI=0 (避免窗边界信息泄漏, 设计文档 §7.5 test_dhi_first_step_zero)
            N' = Σ_d max(0, ⌊(T_d - L) / stride⌋ + 1)
        hi_end: (N',) float32  窗末 HI 值 (供 HI bin 分箱 / 监督标签)
    """
    import pandas as pd
    hi = np.asarray(hi, dtype=np.float32)
    t_index = np.asarray(t_index)
    df = pd.DataFrame({"hi": hi, "bid": list(device_ids), "t": t_index})
    x_list, end_list = [], []
    for _, g in df.groupby("bid", sort=False):
        g = g.sort_values("t")
        h = g["hi"].to_numpy().astype(np.float32)
        T = len(h)
        if T < L:
            continue
        for s in range(0, T - L + 1, stride):
            win_hi = h[s:s + L]
            # 窗级 ΔHI: np.diff 前向差分 (长 L-1), 前补 0 保窗长 L
            # (设计文档 §3.2 "首位补 0 保形状"; §7.5 窗首位 ΔHI=0 防边界泄漏)
            win_d = np.zeros(L, dtype=np.float32)
            win_d[1:] = np.diff(win_hi)
            x_list.append(np.stack([win_hi, win_d], axis=-1))
            end_list.append(float(h[s + L - 1]))
    if not x_list:
        return (np.zeros((0, L, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    x_HI = np.stack(x_list, axis=0).astype(np.float32)
    hi_end = np.array(end_list, dtype=np.float32)
    return x_HI, hi_end
