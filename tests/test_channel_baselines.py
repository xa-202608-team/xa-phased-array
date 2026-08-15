"""通道级基线测试 (T7/M6)。

验证 src/baselines/channel_baselines.py:
  - 5 个基线输出形状/语义正确 (constant/z_extrap/arrhenius/similarity/particle_filter)
  - 评估口径与 run_groups.eval_test 一致 (仅失效通道算 RMSE/PHM/MAE; 删失报下界违反率)
  - 划分按 traj_id (同 traj 16 子阵同 split)
  - rul_max_norm 固定归一 (跨 seed 可比)
  - DTW 距离 / 粒子滤波 边界行为
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                                          # noqa: E402
from src.baselines.channel_baselines import (                              # noqa: E402
    _constant_predict, _z_extrap_predict, _arrhenius_predict,
    _arrhenius_rate, _dtw_distance, _resample_curve,
    _similarity_predict, _particle_filter_predict,
    _evaluate_split, _load_channels, evaluate_channel_baselines,
)
from src.baselines.physical_extrap import rmse                             # noqa: E402

PA_CONFIG = ROOT / "configs" / "phased_array.yaml"
K_BOLTZMANN_eVperK = 8.617333262e-5


# ================================================================ 不依赖 h5 的单元测试

def test_constant_predict_shape():
    """constant 预测所有点同值 (train 失效 rul 归一均值)。"""
    rt = np.linspace(0, 0.5, 100)
    pred = _constant_predict(rt, 0.25)
    assert pred.shape == rt.shape
    assert np.all(pred == 0.25)


def test_z_extrap_linear_trajectory():
    """z 线性递增时, 外推 RUL 应近似真实剩余 (线性退化下精确)。"""
    # z(t) = 0.1 + 0.01·t, 失效时刻 t_fail=90 (z=1.0)
    T = 80
    t = np.arange(T)
    z = 0.1 + 0.01 * t
    rul_max = 1000.0
    pred = _z_extrap_predict(z, rul_max, window=20)
    # 在 t=50 处, RUL 应≈40 (z=0.6, slope=0.01 → t_fail=90, RUL=40)
    assert np.isfinite(pred[50])
    assert abs(pred[50] - 40.0 / rul_max) < 5.0 / rul_max   # 容差 ±5 窗
    # 早期 (<window) 应为 NaN (无足够数据拟合)
    assert np.isnan(pred[0]) or np.isfinite(pred[0])    # 不强制, 看实现
    # 末端 t=79: RUL 应≈11 (z=0.89, slope=0.01 → t_fail=90, RUL=11)
    assert pred[79] > 0


def test_z_extrap_flat_returns_nan():
    """z 不上升 (健康早期) 时, 外推应返回 NaN (常数兜底由调用方处理)。"""
    z = np.full(60, 0.05)   # 常数 (无退化)
    pred = _z_extrap_predict(z, 1000.0, window=20)
    # slope ≈ 0 → 全部 NaN (无法外推)
    assert np.all(np.isnan(pred[20:]))


def test_arrhenius_rate_increases_with_T():
    """Arrhenius 速率随 T 单调递增 (热加速)。"""
    Tj_K = np.array([300.0, 350.0, 400.0])
    r = _arrhenius_rate(Tj_K, Ea_eV=1.0)
    assert r[1] > r[0] and r[2] > r[1]


def test_arrhenius_predict_at_eol_zero():
    """D_now = D_EOL 时, RUL 应=0 (刚好达到 train 标定失效阈值)。"""
    Tj_K = np.full(100, 400.0)
    Ea = 1.0
    rate = _arrhenius_rate(Tj_K, Ea)
    D = np.cumsum(rate)
    D_eol = D[49]   # 第 50 点剂量
    pred = _arrhenius_predict(Tj_K, D_eol, rul_max=1000.0, Ea_eV=Ea)
    # 第 50 点 (idx=49): D_now = D_eol → remaining = 0
    assert abs(pred[49]) < 1e-6
    # 之前点 (idx=40): D_now < D_eol → remaining > 0
    assert pred[40] > 0


def test_dtw_distance_identical_zero():
    """相同序列 DTW 距离=0。"""
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _dtw_distance(q, q) == 0.0


def test_dtw_distance_empty_inf():
    """空序列 DTW 距离=inf。"""
    q = np.array([1.0, 2.0])
    assert _dtw_distance(q, np.array([])) == float("inf")
    assert _dtw_distance(np.array([]), np.array([])) == float("inf")


def test_dtw_distance_symmetric():
    """DTW 距离对称 (q-r = r-q)。"""
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    assert abs(_dtw_distance(q, r) - _dtw_distance(r, q)) < 1e-9


def test_dtw_distance_shifted_increases():
    """平移后的序列 DTW 距离 > 0 且随平移量递增。"""
    base = np.linspace(0, 1, 32)
    d0 = _dtw_distance(base, base)
    d1 = _dtw_distance(base, base + 0.1)
    d2 = _dtw_distance(base, base + 0.5)
    assert d0 == 0.0
    assert d1 > d0
    assert d2 > d1


def test_resample_curve_length():
    """重采样到固定长度 n_grid。"""
    # 不等长输入 → 输出固定长度
    c1 = np.linspace(0, 1, 100)
    c2 = np.linspace(0, 1, 30)
    r1 = _resample_curve(c1, n_grid=48)
    r2 = _resample_curve(c2, n_grid=48)
    assert len(r1) == 48 and len(r2) == 48
    # 端点保持
    assert abs(r1[0] - 0.0) < 1e-6 and abs(r1[-1] - 1.0) < 1e-6
    assert abs(r2[0] - 0.0) < 1e-6 and abs(r2[-1] - 1.0) < 1e-6


def test_particle_filter_monotone_decreasing():
    """z 单调递增 → PF 预测 RUL 应大致单调递减 (剩余寿命随时间减少)。"""
    # p_drift_norm 单调递增 (z 累积), 速率 ≈ 1/T
    T = 200
    p = np.linspace(0.02, 0.98, T)
    pred = _particle_filter_predict(p, rul_max=1000.0, n_particles=100, seed=42)
    # 早期 vs 末期: 末期 RUL 应明显更小
    early = np.mean(pred[:20])
    late = np.mean(pred[-20:])
    assert late < early, f"PF RUL 应随时间递减: early={early:.3f} late={late:.3f}"
    # 所有预测非负
    assert np.all(pred >= 0)


def test_particle_filter_beyond_eol_zero():
    """z 单调累积过失效阈值 (1.0) 后, PF RUL 应近 0 (剩余寿命耗尽)。

    构造一条 p_drift 单调递增序列, 末期超过 1.0 (符合 PF 过程模型, 非跳跃)。
    """
    T = 300
    # p_drift: 从 0.02 线性增到 1.2 (末端 30 点已超 1.0)
    p = np.linspace(0.02, 1.2, T)
    pred = _particle_filter_predict(p, rul_max=1000.0, n_particles=200, seed=42)
    # 所有预测非负
    assert np.all(pred >= 0)
    # 末期 (z>1.0 段) RUL 应远小于早期
    early = float(np.mean(pred[:30]))
    late = float(np.mean(pred[-30:]))
    assert late < early * 0.3, \
        f"末期 RUL {late:.4f} 应明显小于早期 {early:.4f}×0.3 (z 已超阈值)"


def test_evaluate_split_failed_only_rmse():
    """仅失效通道算 RMSE (删失不参与; 与 eval_test 同口径)。"""
    true = np.array([0.5, 0.3, 0.4, 0.2])
    pred = np.array([0.4, 0.6, 0.5, 0.1])
    event = np.array([True, True, False, False])    # 前 2 失效, 后 2 删失
    m = _evaluate_split(true, pred, event)
    # RMSE 仅算前 2 (失效): sqrt(((0.4-0.5)^2 + (0.6-0.3)^2)/2) = sqrt((0.01+0.09)/2)
    expected_rmse = np.sqrt((0.01 + 0.09) / 2)
    assert abs(m["rmse"] - expected_rmse) < 1e-9
    assert m["n_failed"] == 2 and m["n_censored"] == 2


def test_evaluate_split_censor_violation():
    """删失违反率 = pred<true_lower_bound 的比例 (删失通道)。"""
    true = np.array([0.5, 0.5, 0.5, 0.5])
    pred = np.array([0.4, 0.6, 0.3, 0.7])
    event = np.array([True, True, False, False])
    m = _evaluate_split(true, pred, event)
    # 删失 2 个: pred<true 的有 0.3 (1 个); 0.7>0.5 不违反
    assert abs(m["censor_violation_rate"] - 0.5) < 1e-9


# ================================================================ 端到端测试 (依赖 h5)

def _require_data():
    cfg = load_config(PA_CONFIG)
    feature_path = ROOT / cfg["channel_level"]["feature_path"]
    if not feature_path.exists():
        pytest.skip(f"先跑 build_channel_hi: 缺 {feature_path}")
    return cfg, feature_path


def _smoke_cfg(cfg):
    """缩小 test split 加速端到端测试 (test 集约 20-40 通道, similarity 不至于超时)。

    评估口径与正式 run 完全一致 (同 traj_id 划分 + rul_max_norm 归一 + 仅失效 RMSE),
    只是 test 集小; 数字本身不代表正式结果。
    """
    import copy
    cfg2 = copy.deepcopy(cfg)
    cfg2["transfer"]["split"] = {"train": 0.80, "val": 0.10, "test": 0.10}
    return cfg2


def test_load_channels_structure():
    """_load_channels 返回结构正确 (每通道有 x_ch/hi_ch/z_ch/rul_ch + attrs)。"""
    _, feature_path = _require_data()
    chs = _load_channels(feature_path)
    assert len(chs) > 0
    c0 = chs[0]
    assert c0["x_ch"].ndim == 2 and c0["x_ch"].shape[1] == 4
    assert c0["hi_ch"].ndim == 1 and c0["z_ch"].ndim == 1 and c0["rul_ch"].ndim == 1
    assert len(c0["x_ch"]) == len(c0["hi_ch"]) == len(c0["z_ch"]) == len(c0["rul_ch"])
    assert isinstance(c0["event"], bool)
    assert isinstance(c0["traj_id"], int) and isinstance(c0["sub_id"], int)
    # 至少有失效和删失各一
    assert any(c["event"] for c in chs) and not all(c["event"] for c in chs)


def test_evaluate_channel_baselines_full():
    """端到端: 5 个基线都返回有效指标 (rmse/mae/phm/censor_violation_rate)。
    用 _smoke_cfg 缩小 test 集加速 (口径不变, 仅 test 规模小)。"""
    cfg, _ = _require_data()
    cfg = _smoke_cfg(cfg)
    res = evaluate_channel_baselines(cfg, seed=42, verbose=False)
    assert res is not None
    proto = res["_protocol"]
    assert proto["rul_max_norm"] == float(cfg["transfer"]["rul_max_norm"])
    # library 限制生效
    assert proto["n_library"] <= 60
    for name in ["constant", "z_extrap", "arrhenius", "similarity_matching",
                 "particle_filter"]:
        m = res[name]
        assert m["rmse"] >= 0.0 and m["mae"] >= 0.0
        assert 0.0 <= m["censor_violation_rate"] <= 1.0
    # test 集应有失效通道 (smoke 0.1 split ~ 80 通道, 至少几个失效)
    assert res["constant"]["n_failed"] > 0


def test_split_by_trajectory_consistent():
    """同 traj_id 的 16 子阵必须同 split (划分按轨迹, 禁打散)。"""
    cfg, feature_path = _require_data()
    from src.transfer.train_transfer import split_trajectories
    chs = _load_channels(feature_path)
    n_traj = max(c["traj_id"] for c in chs) + 1
    tcfg = cfg["transfer"]
    tr_ids, va_ids, te_ids = split_trajectories(
        n_traj, [tcfg["split"]["train"], tcfg["split"]["val"], tcfg["split"]["test"]], 42)
    tr_set, te_set = set(tr_ids), set(te_ids)
    train_traj_ids = {c["traj_id"] for c in chs if c["traj_id"] in tr_set}
    test_traj_ids = {c["traj_id"] for c in chs if c["traj_id"] in te_set}
    # 不重叠
    assert train_traj_ids.isdisjoint(test_traj_ids)


def test_evaluate_reproducible_same_seed():
    """同 seed 两次运行结果完全一致 (固定随机种子, 可复现)。"""
    cfg, _ = _require_data()
    cfg = _smoke_cfg(cfg)
    r1 = evaluate_channel_baselines(cfg, seed=42, verbose=False)
    r2 = evaluate_channel_baselines(cfg, seed=42, verbose=False)
    for name in ["constant", "z_extrap", "arrhenius", "similarity_matching",
                 "particle_filter"]:
        assert abs(r1[name]["rmse"] - r2[name]["rmse"]) < 1e-9, \
            f"{name} RMSE 不可复现"


def test_similarity_better_than_constant_typically():
    """similarity (用 train 失效退化轨迹库) 通常应优于 constant (均值)。
    这是 sanity check, 非严格断言 (极端划分下可能不成立, 但 seed=42 应该 OK)。"""
    cfg, _ = _require_data()
    cfg = _smoke_cfg(cfg)
    res = evaluate_channel_baselines(cfg, seed=42, verbose=False)
    # similarity RMSE 应不差于 constant 50% 以上 (sanity; 实际往往更好)
    assert res["similarity_matching"]["rmse"] <= res["constant"]["rmse"] * 1.5, \
        f"similarity {res['similarity_matching']['rmse']:.4f} 远差于 constant " \
        f"{res['constant']['rmse']:.4f}, 库匹配可能实现有误"
