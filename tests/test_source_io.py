"""source_io 统一 reader 测试: schema_v1 扁平 + schema_v2 分组双兼容 + LOO 不泄漏。

验证 Task 5 的 reader 能同时读飞轮 (v1) 和相控阵真实/合成 (v2) 两种 h5 结构,
LOO 划分严格按器件, 不发生时间窗泄漏。
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.preprocess.source_io import load_source_features       # noqa: E402

SCHEMA_V2_REAL = ROOT / "data/features/phased_array/schema_v2/source/mosfet_source_features.h5"
SCHEMA_V1_WHEEL = ROOT / "data/features/wheel/schema_v1/source_features.h5"


def test_read_schema_v2_real_mosfet():
    """真实 NASA MOSFET h5 (schema_v2 分组) 能正确读取为扁平视图。"""
    if not SCHEMA_V2_REAL.exists():
        pytest.skip("真实 schema_v2 MOSFET h5 不存在 (需先跑 mosfet_real_loader)")
    sd = load_source_features(SCHEMA_V2_REAL, id_field="device_id")
    assert sd.schema_version == "2"
    assert sd.n_features == 2                               # [RDS_drift, T_case_C]
    assert sd.feature_names == ["RDS_drift", "T_case_C"]
    assert sd.n_devices >= 30                               # 真实 37 case
    N = sd.features.shape[0]
    assert sd.features.shape == (N, 2)
    assert sd.hi.shape == (N,) and sd.rul.shape == (N,)
    assert len(sd.device_ids) == N and len(sd.split) == N
    # LOO: train/val 器件不交集
    train_devs = {d for d, s in zip(sd.device_ids, sd.split) if s == "train"}
    val_devs = {d for d, s in zip(sd.device_ids, sd.split) if s == "val"}
    assert len(train_devs & val_devs) == 0
    assert len(val_devs) >= 1                               # 至少 1 个 val 器件
    # 真实数据 RUL 为秒单位 (量级 ~10^4), rul_max 暴露给调用方归一
    assert sd.rul_max > 1.0


def test_read_schema_v1_wheel():
    """飞轮 wheel h5 (schema_v1 扁平) 不受 v2 改动影响。"""
    if not SCHEMA_V1_WHEEL.exists():
        pytest.skip("飞轮 schema_v1 source h5 不存在 (需先跑 wheel_features)")
    sd = load_source_features(SCHEMA_V1_WHEEL, id_field="bearing_id")
    assert sd.schema_version == "1"
    assert sd.n_devices >= 1
    assert set(sd.split) <= {"train", "val"}                # v1 沿用 h5 内 split 字段


def test_val_device_ids_override():
    """val_device_ids 显式指定时, 对应器件标 val, 其余 train。"""
    if not SCHEMA_V2_REAL.exists():
        pytest.skip("真实 schema_v2 MOSFET h5 不存在")
    sd = load_source_features(SCHEMA_V2_REAL, id_field="device_id",
                              val_device_ids=["Test_1"])
    val_devs = {d for d, s in zip(sd.device_ids, sd.split) if s == "val"}
    assert val_devs == {"Test_1"}


def test_unknown_schema_raises():
    """无法识别的 h5 结构应抛 ValueError (清晰报错, 非神秘 KeyError)。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("weird", data=np.zeros(3))
        with pytest.raises(ValueError, match="schema"):
            load_source_features(p)


def test_event_observed_and_lower_bound_v2():
    """D1 修复: v2 真实 h5 读出 event_observed / rul_lower_bound 逐行字段。

    删失器件 (event_observed=False) 应有非零 lower_bound (censor_time - t),
    且 hi 平均低 (健康器件); 失效器件 lb == rul (精确 RUL)。
    """
    if not SCHEMA_V2_REAL.exists():
        pytest.skip("真实 schema_v2 MOSFET h5 不存在")
    sd = load_source_features(SCHEMA_V2_REAL, id_field="device_id")
    N = sd.features.shape[0]
    assert sd.event_observed.shape == (N,)
    assert sd.event_observed.dtype == bool
    assert sd.rul_lower_bound.shape == (N,)
    # 真实数据有失效 + 删失两种器件
    n_failed_devs = len({d for d, e in zip(sd.device_ids, sd.event_observed) if e})
    n_censored_devs = len({d for d, e in zip(sd.device_ids, sd.event_observed) if not e})
    assert n_failed_devs >= 20                                   # 真实 29 失效
    assert n_censored_devs >= 5                                  # 真实 8 删失
    # 删失器件行的 lower_bound 不全 0 (censor_time - t, 早期 t 小 lb 大)
    cens_rows = ~sd.event_observed
    assert cens_rows.sum() >= 1000
    assert sd.rul_lower_bound[cens_rows].max() > 0.1             # 早期 lb 远大于 0
    # 删失器件 HI 平均低于失效器件 (健康 vs 退化)
    cens_hi_mean = float(sd.hi[cens_rows].mean())
    fail_hi_mean = float(sd.hi[sd.event_observed].mean())
    assert cens_hi_mean < fail_hi_mean


def test_hi_isotonic_denoise_v2():
    """D6 修复: v2 hi 默认做 isotonic 去噪, 单调违规率应大幅下降。"""
    if not SCHEMA_V2_REAL.exists():
        pytest.skip("真实 schema_v2 MOSFET h5 不存在")
    sd_iso = load_source_features(SCHEMA_V2_REAL, id_field="device_id", hi_isotonic=True)
    sd_raw = load_source_features(SCHEMA_V2_REAL, id_field="device_id", hi_isotonic=False)
    # 原始 hi 含噪非单调 (~15%)
    d_raw = np.diff(sd_raw.hi)
    viol_raw = float((d_raw < -1e-6).mean())
    # isotonic 后应接近 0
    d_iso = np.diff(sd_iso.hi)
    viol_iso = float((d_iso < -1e-6).mean())
    assert viol_iso < 0.01, f"isotonic 后 mono_viol={viol_iso:.4f} 仍高"
    assert viol_iso < viol_raw * 0.1, f"改善不足: raw={viol_raw:.4f} iso={viol_iso:.4f}"


def test_v1_event_observed_defaults_true():
    """v1 (飞轮 wheel) 无 event_observed 字段, 默认全 True (真实失效轨迹)。"""
    if not SCHEMA_V1_WHEEL.exists():
        pytest.skip("飞轮 schema_v1 source h5 不存在")
    sd = load_source_features(SCHEMA_V1_WHEEL, id_field="bearing_id")
    assert sd.event_observed.all()
    # lb == rul (v1 无删失概念)
    np.testing.assert_allclose(sd.rul_lower_bound, sd.rul, rtol=0, atol=1e-6)


def test_rul_loss_hinge_on_censored():
    """D1: _rul_loss 对删失行用 hinge (只罚 pred<lb), 失效行用 Huber。"""
    import torch
    import torch.nn as nn
    from src.train.pretrain import _rul_loss
    huber = nn.HuberLoss(delta=1.0, reduction="none")
    # 构造 4 行: 2 失效 + 2 删失 (shape 一致 (4,))
    pred = torch.tensor([0.5, 0.0, 0.8, 0.1], dtype=torch.float32)
    label = torch.tensor([0.5, 0.0, 0.8, 0.1], dtype=torch.float32)  # 失效行=精确; 删失行=lb (填充)
    ev = torch.tensor([True, True, False, False])
    lb = torch.tensor([0.5, 0.0, 0.3, 0.2], dtype=torch.float32)  # 删失行 lb
    # 失效行 0: pred=0.5, label=0.5 → huber=0
    # 失效行 1: pred=0.0, label=0.0 → huber=0
    # 删失行 0: pred=0.8 > lb=0.3 → hinge=0 (允许大预测)
    # 删失行 1: pred=0.1 < lb=0.2 → hinge=(0.2-0.1)^2=0.01
    loss, L_O, L_C = _rul_loss(pred, label, ev, lb, huber, eta=1.0)
    # L_O = mean(0, 0) = 0
    assert float(L_O) < 1e-6
    # L_C = mean(0, 0.01) = 0.005
    assert abs(float(L_C) - 0.005) < 1e-5
    # eta=2 时总 loss 翻倍 (L_C 本身不变)
    loss2, _, L_C2 = _rul_loss(pred, label, ev, lb, huber, eta=2.0)
    assert abs(float(L_C2) - 0.005) < 1e-5                      # L_C 本身不变
    assert abs(float(loss2) - (float(L_O) + 2.0 * float(L_C))) < 1e-5


# ----------------------------------------------------------------
# make_hi_windows 单元测试 (设计文档 §3.2/§4.5)
# 不依赖 HISeqEncoder, 纯 numpy 数据侧工具。
# ----------------------------------------------------------------
def test_make_hi_windows_shape_and_channels():
    """形状 (N, L, 2); 通道 0=HI, 通道 1=ΔHI 首位补 0; hi_end=窗末 HI。

    两器件 (A=10 步线性, B=8 步线性), L=4, stride=2:
      窗数 A=floor((10-4)/2)+1=4, B=floor((8-4)/2)+1=3, 共 7
    """
    from src.data.preprocess.source_io import make_hi_windows
    hi_A = np.linspace(0.0, 0.9, 10, dtype=np.float32)
    hi_B = np.linspace(0.5, 0.8, 8, dtype=np.float32)
    hi = np.concatenate([hi_A, hi_B])
    ids = ["A"] * 10 + ["B"] * 8
    t = np.concatenate([np.arange(10), np.arange(8)])
    L, stride = 4, 2
    x_HI, hi_end = make_hi_windows(hi, ids, t, L, stride)
    # 形状
    assert x_HI.shape == (7, L, 2)
    assert x_HI.dtype == np.float32
    assert hi_end.shape == (7,)
    assert hi_end.dtype == np.float32
    # 通道 0 = HI 序列 (A 首窗 = hi_A[0:4])
    np.testing.assert_allclose(x_HI[0, :, 0], hi_A[:4], rtol=1e-6)
    # 通道 1 = ΔHI, 首位补 0 (每窗第一步)
    np.testing.assert_allclose(x_HI[:, 0, 1], 0.0, atol=1e-7)
    # A 器件 hi 等差 0.1, 故 ΔHI 窗内 [1:] = 0.1
    np.testing.assert_allclose(x_HI[0, 1:, 1], 0.1, rtol=1e-5)
    # hi_end = 窗末 HI (A 首窗末 = hi_A[3] = 0.3)
    np.testing.assert_allclose(hi_end[0], hi_A[3], rtol=1e-6)
    # A 器件 4 窗末值 = [hi_A[3], hi_A[5], hi_A[7], hi_A[9]]
    np.testing.assert_allclose(hi_end[:4], hi_A[[3, 5, 7, 9]], rtol=1e-6)


def test_make_hi_windows_no_cross_device():
    """多器件 groupby 严格不串器件: 窗内 HI 必须来自同一器件。

    A=[0,0,0,0,0,1.0] (末步突变), B=[0.5]*6 (常量); L=3 stride=1 → 每器件 4 窗。
    """
    from src.data.preprocess.source_io import make_hi_windows
    hi_A = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    hi_B = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    hi = np.concatenate([hi_A, hi_B])
    ids = ["A"] * 6 + ["B"] * 6
    t = np.concatenate([np.arange(6), np.arange(6)])
    x_HI, hi_end = make_hi_windows(hi, ids, t, L=3, stride=1)
    assert x_HI.shape == (8, 3, 2)
    # 前 4 窗属 A, 第 4 窗 (s=3) 末值 = hi_A[5] = 1.0
    assert x_HI[3, -1, 0] == 1.0
    # 后 4 窗属 B, HI 通道全 = 0.5
    np.testing.assert_allclose(x_HI[4:, :, 0], 0.5, rtol=1e-6)
    # B 常量序列 → ΔHI 全 0
    np.testing.assert_allclose(x_HI[4:, :, 1], 0.0, atol=1e-7)
    # hi_end 前 4 = A 窗末, 后 4 = 0.5
    np.testing.assert_allclose(hi_end[4:], 0.5, rtol=1e-6)


def test_make_hi_windows_stride_and_sort():
    """stride 控窗数; t_index 乱序输入应按 t 排序后滑窗。"""
    from src.data.preprocess.source_io import make_hi_windows
    # 单器件 5 步线性 hi = [0.0, 0.25, 0.5, 0.75, 1.0]
    hi = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    # 故意打乱 t_index 顺序
    perm = np.array([3, 1, 4, 0, 2])
    hi_shuffled = hi[perm]
    t_shuffled = perm.copy()       # 用 t 来还原顺序
    ids = ["X"] * 5
    # L=3, stride=1: 排序后 T=5 → 3 窗 (s=0,1,2)
    x_HI, hi_end = make_hi_windows(hi_shuffled, ids, t_shuffled, L=3, stride=1)
    assert x_HI.shape == (3, 3, 2)
    # 排序后首窗 = hi[0:3] = [0.0, 0.25, 0.5]
    np.testing.assert_allclose(x_HI[0, :, 0], [0.0, 0.25, 0.5], rtol=1e-6)
    # stride=2 控窗数: T=5, L=3, stride=2 → 2 窗 (s=0, 2)
    x_HI2, hi_end2 = make_hi_windows(hi_shuffled, ids, t_shuffled, L=3, stride=2)
    assert x_HI2.shape == (2, 3, 2)
    np.testing.assert_allclose(hi_end2, [0.5, 1.0], rtol=1e-6)


def test_make_hi_windows_short_device():
    """短器件 (T<L) 不产窗, 返回空数组不报错。"""
    from src.data.preprocess.source_io import make_hi_windows
    hi = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    ids = ["short"] * 3                        # 长度与 hi 对齐
    t = np.arange(3)
    x_HI, hi_end = make_hi_windows(hi, ids, t, L=5, stride=1)
    assert x_HI.shape == (0, 5, 2)
    assert hi_end.shape == (0,)
    # 全短器件 + 一个长器件混合
    hi2 = np.concatenate([hi, np.linspace(0.0, 1.0, 8, dtype=np.float32)])
    ids2 = ["short"] * 3 + ["long"] * 8
    t2 = np.concatenate([np.arange(3), np.arange(8)])
    x_HI2, hi_end2 = make_hi_windows(hi2, ids2, t2, L=5, stride=1)
    # 只 long 器件产窗: 8-5+1=4
    assert x_HI2.shape == (4, 5, 2)
    assert hi_end2.shape == (4,)
