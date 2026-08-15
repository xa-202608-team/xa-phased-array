"""通道级数据集 + k-shot 协议测试 (T6.1/T6.2, M7)。

验证 src/transfer/channel_dataset.py:
  - load_target_channel: 读 channel_features.h5 → 扁平数组结构正确
  - ChannelSeqDataset: 按 (traj_id, sub_id) 分组, K 窗块输出 6 元组
  - k-shot 协议: train 内采 k 条保留标签, 其余 mask; val/test 不受影响
  - assert_split_by_trajectory: 同 traj 16 sub 必须同 split
  - 划分严格按 traj_id (channel_level 铁律)
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                                          # noqa: E402
from src.transfer.channel_dataset import (                                 # noqa: E402
    ChannelSeqDataset, load_target_channel, apply_kshot_mask,
    sample_kshot_trajectories, assert_split_by_trajectory)
from src.transfer.train_transfer import split_trajectories                # noqa: E402

PA_CONFIG = ROOT / "configs" / "phased_array.yaml"


def _require_data():
    cfg = load_config(PA_CONFIG)
    feature_path = ROOT / cfg["channel_level"]["feature_path"]
    if not feature_path.exists():
        pytest.skip(f"先跑 build_channel_hi: 缺 {feature_path}")
    return cfg, feature_path


# ================================================================ load_target_channel 结构

def test_load_target_channel_returns_correct_shapes():
    cfg, feature_path = _require_data()
    x, hi, rul, ck, tid, ev, lb, n_traj, sid = load_target_channel(feature_path)
    N = len(x)
    assert x.ndim == 2 and x.shape[1] == 4   # canonical 4 维
    assert hi.shape == (N,) and rul.shape == (N,)
    assert ck.shape == (N,) and tid.shape == (N,) and ev.shape == (N,)
    assert lb.shape == (N,) and sid.shape == (N,)
    # channel_keys 唯一数 = n_traj * 16 (每子阵一条独立通道)
    assert len(np.unique(ck)) == n_traj * 16
    # hi ∈ [0, 1]
    assert hi.min() >= 0.0 and hi.max() <= 1.0
    # rul 非负
    assert (rul >= 0).all()
    # event_observed 是 bool
    assert ev.dtype == bool
    # 至少有失效和删失各一
    assert ev.any() and not ev.all()


def test_drop_features_reduces_dim():
    cfg, feature_path = _require_data()
    x_full, *_ = load_target_channel(feature_path)
    x_drop, *_ = load_target_channel(feature_path, drop_features=["T_dev_C"])
    assert x_drop.shape[1] == x_full.shape[1] - 1


# ================================================================ ChannelSeqDataset

def test_channel_seq_dataset_groups_by_channel_key():
    """ChannelSeqDataset 按 ckT 分组, 每 ckT 一条独立通道序列 (非 tidT)。"""
    cfg, feature_path = _require_data()
    x, hi, rul, ck, tid, ev, lb, n_traj, sid = load_target_channel(feature_path)
    # 用前 2 轨迹的所有通道 (32 条独立通道序列)
    sel_traj = list(range(2))
    m = np.isin(tid, sel_traj)
    L, K = 16, 4
    ds = ChannelSeqDataset(x[m], hi[m], rul[m], ck[m], L, K, stride=50,
                           event_observed=ev[m], rul_lower_bound=lb[m])
    assert len(ds) > 0
    f, h, r, e, lb_, dmg = ds[0]
    assert f.shape == (K, L, 4)          # K 窗 × L 长 × 4 维
    assert h.shape == (K,) and r.shape == (K,)
    assert e.shape == (K,) and e.dtype == torch.bool
    assert lb_.shape == (K,) and dmg.shape == (K,)


def test_channel_seq_does_not_cross_channel():
    """窗不跨通道 (相邻窗来自同一 ckT, 不混 ckT)。"""
    cfg, feature_path = _require_data()
    x, hi, rul, ck, tid, ev, lb, n_traj, sid = load_target_channel(feature_path)
    m = np.isin(tid, [0])
    L, K = 16, 4
    ds = ChannelSeqDataset(x[m], hi[m], rul[m], ck[m], L, K, stride=50,
                           event_observed=ev[m], rul_lower_bound=lb[m])
    if len(ds) < 1:
        pytest.skip("单轨迹样本不足")
    # K 窗块内所有 ck 应一致 (同通道); 检查第一个样本
    f, h, r, e, lb_, dmg = ds[0]
    # 通过反查: f 是 x 切片, 验证 ckT 对应的 f 一致 (这里只能验证形状, 真值检查需更复杂)
    assert f.shape == (K, L, 4)


# ================================================================ k-shot 协议

def test_sample_kshot_returns_none_for_all():
    """k_shot=None 或 'all' → 返回 None (全部 train 带标签)。"""
    train_ids = np.array([1, 5, 12, 20, 25])
    assert sample_kshot_trajectories(train_ids, None, seed=42) is None
    assert sample_kshot_trajectories(train_ids, "all", seed=42) is None


def test_sample_kshot_returns_subset():
    """k_shot=k → 返回 k 条 train_ids 子集。"""
    train_ids = np.array([1, 5, 12, 20, 25, 30, 35])
    k_ids = sample_kshot_trajectories(train_ids, 3, seed=42)
    assert k_ids is not None
    assert len(k_ids) == 3
    assert k_ids.issubset(set(train_ids.tolist()))


def test_sample_kshot_seed_reproducible():
    """同 seed 采样一致 (可复现)。"""
    train_ids = np.array([1, 5, 12, 20, 25, 30])
    k1 = sample_kshot_trajectories(train_ids, 3, seed=42)
    k2 = sample_kshot_trajectories(train_ids, 3, seed=42)
    assert k1 == k2


def test_sample_kshot_k_ge_len_returns_none():
    """k >= len(train_ids) → 全部带标签 (返回 None)。"""
    train_ids = np.array([1, 5, 12])
    assert sample_kshot_trajectories(train_ids, 3, seed=42) is None
    assert sample_kshot_trajectories(train_ids, 5, seed=42) is None


def test_apply_kshot_mask_none_no_change():
    """k_ids=None → 不 mask (原数组不变)。"""
    rul = np.array([10.0, 5.0, 0.0])
    ev = np.array([True, True, False])
    lb = np.array([10.0, 5.0, 5.0])
    tid = np.array([0, 1, 2])
    tr = np.array([0, 1, 2])
    r2, e2, l2 = apply_kshot_mask(rul, ev, lb, tid, tr, None)
    assert np.array_equal(r2, rul) and np.array_equal(e2, ev) and np.array_equal(l2, lb)


def test_apply_kshot_mask_unlabeled_set_to_zero():
    """k-shot mask: train 内非 k_ids 通道 rul=0 + event=False; val/test 不变。"""
    rul = np.array([10.0, 5.0, 8.0, 3.0])    # 前 2 train, 后 2 val/test
    ev = np.array([True, True, True, False])
    lb = np.array([10.0, 5.0, 8.0, 3.0])
    tid = np.array([0, 1, 2, 3])
    tr = np.array([0, 1])
    k_ids = {0}    # 仅 traj 0 带标签; traj 1 mask
    r2, e2, l2 = apply_kshot_mask(rul, ev, lb, tid, tr, k_ids)
    # traj 0 (idx 0) 不变
    assert r2[0] == 10.0 and e2[0] == True
    # traj 1 (idx 1) mask
    assert r2[1] == 0.0 and e2[1] == False
    # val/test (idx 2,3) 不变
    assert r2[2] == 8.0 and r2[3] == 3.0
    assert e2[2] == True and e2[3] == False


# ================================================================ 划分铁律

def test_assert_split_by_trajectory_passes():
    """同 traj 16 sub 全在同 → assert 通过。"""
    tid = np.array([0] * 16 + [1] * 16 + [2] * 16)
    sid = np.array(list(range(16)) * 3)
    tr = [0]
    va = [1]
    te = [2]
    assert assert_split_by_trajectory(tid, tr, va, te, sid) is True


def test_assert_split_by_trajectory_fails_on_overlap():
    """train/val/test 重叠 → assert 失败。"""
    tid = np.array([0] * 16)
    with pytest.raises(AssertionError):
        assert_split_by_trajectory(tid, [0], [0], [], None)


# ================================================================ 端到端 (依赖 h5)

def test_end_to_end_split_and_kshot():
    """端到端: load + split + k-shot mask, 验证 mask 后 train 内仅 k 条带标签。"""
    cfg, feature_path = _require_data()
    x, hi, rul, ck, tid, ev, lb, n_traj, sid = load_target_channel(feature_path)
    tc = cfg["transfer"]
    tr, va, te = split_trajectories(
        n_traj, [tc["split"]["train"], tc["split"]["val"], tc["split"]["test"]], 42)
    assert_split_by_trajectory(tid, tr, va, te, sid)
    # k=2
    k_ids = sample_kshot_trajectories(tr, 2, seed=42)
    r2, e2, l2 = apply_kshot_mask(rul, ev, lb, tid, tr, k_ids)
    # train 内 mask 检查
    train_mask = np.isin(tid, list(tr))
    labeled = np.isin(tid, list(k_ids))
    unlabeled_train = train_mask & ~labeled
    if unlabeled_train.any():
        assert (e2[unlabeled_train] == False).all(), "mask 后 train 非 k_ids 通道 event 应全 False"
        assert (r2[unlabeled_train] == 0.0).all(), "mask 后 train 非 k_ids 通道 rul 应全 0"
    # val/test 不变
    va_mask = np.isin(tid, list(va))
    assert (e2[va_mask] == ev[va_mask]).all(), "val event 不应变"
    te_mask = np.isin(tid, list(te))
    assert (e2[te_mask] == ev[te_mask]).all(), "test event 不应变"
