"""transfer/channel_dataset.py

通道级数据集 + 加载器 (T6.1/M7, channel_level 路线)。

与 train_transfer.TargetSeqDataset 的区别:
  - 分组键: (traj_id, sub_id) 联合 (每子阵一条独立通道序列), 非单一 traj_id
  - 划分: 严格按 traj_id (同 traj 16 子阵同 split, 写 assert)
  - 输入: x_ch (T,4) canonical device schema (非 x_global+x_nodes)
  - 输出: 与 TargetSeqDataset 同 6 元组 (f, h, r, ev, lb, dmg) → 复用 _train_epoch/eval_test

k-shot 协议 (T6.2):
  - k_shot=None: 全部 train 通道保留 RUL 标签
  - k_shot=k: 仅 k 条 train 轨迹(含 16 子阵)保留 RUL 标签; 其余 train 通道标签 mask
    (event=False + rul=lower_bound=0 + hi 不变), 仅作 MMD 无监督对齐用
  - val/test 不受 k_shot 影响 (始终带标签)

不改 TargetSeqDataset/HIWindowDataset (旧路径飞轮/cross_level_transfer 复跑用)。
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 复用 TargetSeqDataset 的 K 窗块输出契约 (6 元组)
# 直接 import 以保证一致性 (单点修改)
from src.transfer.train_transfer import TargetSeqDataset  # noqa: F401  (re-export)


# F1-A: channel label schema 版本与必填 attrs。消费者一律经 read_channel_label_meta
# 校验后按版本读字段, 缺失即 fail (不 warning 继续), 根除尺度静默漂移。
CHANNEL_LABEL_SCHEMA_V1 = "channel_label_v1"
CHANNEL_LABEL_SCHEMA_V2 = "channel_label_v2"
_CHANNEL_META_V2_REQUIRED = (
    "rul_scale_windows", "sample_period_s", "rul_capped", "mission_horizon_windows")


def read_channel_label_meta(h5: "h5py.File") -> dict:
    """强校验 channel_features.h5 顶层 attrs, 返回标签尺度元数据。

    v2 (channel_label_v2, mission_horizon):
        schema/rul_scale_windows(=H)/sample_period_s/rul_capped/mission_horizon_windows
        缺一即 ValueError; 模型标签读 rul_ch_norm (=rul_ch_windows/H), 物理/推理还原
        读 rul_ch_windows × sample_period_s。
    v1 (channel_label_v1, legacy_cap):
        仅校验 schema; 模型标签读 rul_ch (已按 cap_ratio*T 封顶的窗口数, 旧口径),
        归一仍由调用方用 transfer.rul_max_norm (service 级 4088) 处理。
    无 channel_label_schema attr 视为旧产物, 直接报错要求重建。
    """
    schema = str(h5.attrs.get("channel_label_schema", ""))
    if not schema:
        raise ValueError(
            "channel_features.h5 缺 channel_label_schema attr (疑似 2026-08-22 F1-A 之前的"
            "旧产物); 请重建: python -m src.sim.build_channel_hi")
    if schema not in (CHANNEL_LABEL_SCHEMA_V1, CHANNEL_LABEL_SCHEMA_V2):
        raise ValueError(f"未知 channel_label_schema={schema!r}; 期望 v1/v2")
    meta = {"channel_label_schema": schema}
    if schema == CHANNEL_LABEL_SCHEMA_V2:
        for k in _CHANNEL_META_REQUIRED:
            if k not in h5.attrs:
                raise ValueError(f"v2 channel_features.h5 缺必填 attr {k!r}")
        H = float(h5.attrs["rul_scale_windows"])
        sp = float(h5.attrs["sample_period_s"])
        H_alt = float(h5.attrs["mission_horizon_windows"])
        if H <= 0 or sp <= 0:
            raise ValueError(f"rul_scale_windows/sample_period_s 必须为正, 得 H={H} sp={sp}")
        if abs(H - H_alt) > 1e-6:
            raise ValueError(
                f"rul_scale_windows({H}) != mission_horizon_windows({H_alt}); 数据损坏")
        if str(h5.attrs["rul_capped"]).lower() != "false":
            raise ValueError(
                f"v2 要求 rul_capped='false', 得 {h5.attrs.get('rul_capped')!r}")
        meta.update({
            "rul_scale_windows": H, "sample_period_s": sp,
            "rul_capped": False, "mission_horizon_windows": H_alt,
            "t_dev_unit": str(h5.attrs.get("t_dev_unit", "")),
        })
        if meta["t_dev_unit"] != "degC":
            raise ValueError(
                f"v2 canonical 要求 t_dev_unit='degC', 得 {meta['t_dev_unit']!r}; "
                "重建 build_channel_hi (清零重审温度修复后产物)")
    return meta


class ChannelSeqDataset(TargetSeqDataset):
    """通道级窗口数据集 (继承 TargetSeqDataset, 仅改 docstring; 父类逻辑直接可用)。

    输入 x_ch (N,4) 滑窗 -> (L,4); 标签 (hi_end, z_end→用 hi 代, rul_end, event, rul_lb)。
    分组键 = channel_key (=traj_id*N_SUB+sub_id), 划分按 traj_id (在 load_target_channel 内 assert)。
    """


def load_target_channel(h5_path: Path | str, drop_features: list | None = None,
                        n_sub_per_traj: int = 16):
    """读 channel_features.h5 → 通道级扁平数组。

    返回:
      x_ch (N,4), hi (N,), rul (N,), channel_keys (N,), traj_ids (N,),
      event (N,), rul_lb (N,), n_traj, sub_ids (N,)
    其中 N = Σ_traj Σ_sub T_{traj,sub} (失效通道 T=eol+1, 删失 T=full)。

    返回的 rul 已为**模型标签口径**:
      - v2 (channel_label_v2): rul_ch_norm (窗口数 / H, 已归一, run_groups factor=1.0);
      - v1 (channel_label_v1): rul_ch (窗口数, 由 run_groups 按 rul_max_norm 归一)。
    调用方一律不要再除以 H 或 4088 — 尺度由本函数按 schema 锁定 (F1-A 根除双归一)。
    需 H/绝对窗口的消费者 (基线/绘图/推理) 另用 read_channel_label_meta()。

    **k-shot 协议不在本函数做** (需要先 split_trajectories 确定 train 子集);
    调用方 (run_groups) 在 split 后用 sample_kshot_trajectories 采样 + apply_kshot_mask。
    """
    drop_features = drop_features or []
    x_list, hi_list, rul_list, ev_list, lb_list = [], [], [], [], []
    ck_list, tid_list, sid_list = [], [], []
    traj_ids_set: set[int] = set()
    with h5py.File(h5_path, "r") as f:
        # F1-A: 强校验 schema + v2 必填 attrs, 缺失/版本不符直接 fail (不 warning)
        meta = read_channel_label_meta(f)
        schema = meta["channel_label_schema"]
        rul_field = "rul_ch_norm" if schema == CHANNEL_LABEL_SCHEMA_V2 else "rul_ch"
        traj_keys = sorted(f.keys())
        for tk in traj_keys:
            traj_grp = f[tk]
            traj_id = int(tk.split("_")[1])
            traj_ids_set.add(traj_id)
            sub_keys = sorted(k for k in traj_grp.keys() if k.startswith("sub_"))
            for sk in sub_keys:
                sub = traj_grp[sk]
                sub_id = int(sub.attrs["sub_id"])
                event = bool(sub.attrs["event_observed"])
                T = sub["x_ch"].shape[0]
                x_ch = sub["x_ch"][:].astype(np.float32)
                if drop_features:
                    # canonical 4 维: [p_drift_norm, T_dev_C, duty, drive_norm]
                    from src.sim.build_channel_hi import CANONICAL_COLS
                    keep = [i for i, c in enumerate(CANONICAL_COLS) if c not in drop_features]
                    x_ch = x_ch[:, keep]
                hi = sub["hi_ch"][:].astype(np.float32)
                rul = sub[rul_field][:].astype(np.float32)
                ev_arr = np.full(T, event, dtype=bool)
                lb_arr = rul.copy()  # rul_lower_bound = rul (失效精确 / 删失下界)
                x_list.append(x_ch)
                hi_list.append(hi)
                rul_list.append(rul)
                ev_list.append(ev_arr)
                lb_list.append(lb_arr)
                ck_list.append(np.full(T, traj_id * n_sub_per_traj + sub_id))
                tid_list.append(np.full(T, traj_id))
                sid_list.append(np.full(T, sub_id))
    if not x_list:
        raise RuntimeError(f"channel_features.h5 无子阵数据: {h5_path}")
    n_traj = max(traj_ids_set) + 1
    x_ch = np.concatenate(x_list)
    hi = np.concatenate(hi_list)
    rul = np.concatenate(rul_list)
    ev = np.concatenate(ev_list)
    lb = np.concatenate(lb_list)
    ck = np.concatenate(ck_list)
    tid = np.concatenate(tid_list)
    sid = np.concatenate(sid_list)
    return (x_ch, hi, rul, ck, tid, ev, lb, n_traj, sid)


def sample_kshot_trajectories(train_ids: np.ndarray, k_shot, seed: int):
    """从 train_ids 内按 seed 采 k 条完整轨迹 (k-shot 协议)。

    k_shot=None 或 'all': 返回 None (= 全部 train 带标签)
    k_shot=int k: 返回 set(采样 k 条 traj_ids); k ≥ len(train_ids) 时返回 None (= all)
    """
    if k_shot is None or k_shot == "all":
        return None
    k = int(k_shot)
    if k >= len(train_ids):
        return None
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(train_ids), size=k, replace=False)
    return set(chosen.tolist())


def apply_kshot_mask(rul: np.ndarray, event: np.ndarray, rul_lb: np.ndarray,
                     traj_ids: np.ndarray, train_ids: np.ndarray,
                     k_shot_traj_ids: set | None):
    """在 train split 内应用 k-shot mask (调用方在 split 后调用)。

    k_shot_traj_ids=None: 不 mask (全部 train 带标签)
    k_shot_traj_ids=set: train 内仅 k_shot_traj_ids 中的轨迹保留标签,
                        其余 train 通道 event=False + rul=rul_lb=0 (hinge 不激活)

    返回 (rul, event, rul_lb) 三元组 (被 mask 的副本, 原数组不变)。
    """
    rul = rul.copy()
    event = event.copy()
    rul_lb = rul_lb.copy()
    if k_shot_traj_ids is None:
        return rul, event, rul_lb
    train_mask = np.isin(traj_ids, list(train_ids))
    labeled_mask = np.isin(traj_ids, list(k_shot_traj_ids))
    # train 内但不在 k_shot_traj_ids 的通道 → mask
    unlabeled_mask = train_mask & ~labeled_mask
    rul[unlabeled_mask] = 0.0
    rul_lb[unlabeled_mask] = 0.0
    event[unlabeled_mask] = False
    return rul, event, rul_lb


def assert_split_by_trajectory(traj_ids: np.ndarray, train_ids, val_ids, test_ids,
                               sub_ids: np.ndarray | None = None):
    """断言: 同一 traj_id 的所有 16 sub_id 必须同属一个 split (channel_level 铁律)。

    在 load + split 后调用, 验证划分不变量。失败 → 立即停 (实验卫生门)。
    """
    tr, va, te = set(train_ids), set(val_ids), set(test_ids)
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), \
        "train/val/test 轨迹 id 重叠"
    all_split = tr | va | te
    for tid in np.unique(traj_ids):
        assert int(tid) in all_split, f"traj_id {tid} 不在任何 split"
    if sub_ids is not None:
        # 同 traj 的所有 sub 必须落在同一 split (channel_level 硬约束)
        df = pd.DataFrame({"tid": traj_ids, "sid": sub_ids})
        df["split"] = df["tid"].map(lambda t: "tr" if t in tr else ("va" if t in va else "te"))
        n_splits_per_traj = df.groupby("tid")["split"].nunique()
        assert (n_splits_per_traj == 1).all(), \
            f"同 traj 16 子阵跨 split: {n_splits_per_traj[n_splits_per_traj > 1]}"
    return True
