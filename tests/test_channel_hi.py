"""通道级 HI 构建测试 (T2/M2)。

验证 src/sim/build_channel_hi.py 输出正确性 (不依赖 EOL_ch vs EOL_svc 物理关系):
  - hi_ch ∈ [0,1]
  - z_ch 单调非减 (f_s 单调 → z 单调)
  - 失效通道截断到 EOL (rul[eol]=0, 无 EOL 后零标签窗 — P0-1 铁律)
  - 删失通道 rul 封顶 cap_ratio*T (不反推仿真截止时刻)
  - 同 traj_id 的 16 sub_id 可同属一个 split (划分按轨迹)
  - δ 阈值只来自 config (不随实验数据变)
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                              # noqa: E402

PA_CONFIG = ROOT / "configs" / "phased_array.yaml"


def _load_channels():
    cfg = load_config(PA_CONFIG)
    feature_path = ROOT / cfg["channel_level"]["feature_path"]
    if not feature_path.exists():
        pytest.skip(f"先跑 build_channel_hi: 缺 {feature_path}")
    return feature_path, cfg


def _iter_subs(f, n_traj=5):
    """迭代前 n_traj 轨迹的所有子阵组。"""
    for key in sorted(f.keys())[:n_traj]:
        traj = f[key]
        for sub_key in sorted(k for k in traj.keys() if k.startswith("sub_")):
            yield key, sub_key, traj[sub_key]


def test_hi_in_unit_interval():
    feature_path, _ = _load_channels()
    with h5py.File(feature_path, "r") as f:
        for _key, sub_key, sub in _iter_subs(f, n_traj=5):
            hi = sub["hi_ch"][:]
            assert hi.min() >= 0.0 and hi.max() <= 1.0, \
                f"{_key}/{sub_key} hi_ch 超界 [{hi.min()}, {hi.max()}]"


def test_z_monotone_low_violation():
    """z_ch 应单调非减 (f_s=eff_age/life_scale 累积非减 → dR/dI/dg 非减 → z=max 非减)。"""
    feature_path, _ = _load_channels()
    viols = []
    with h5py.File(feature_path, "r") as f:
        for _key, _sub_key, sub in _iter_subs(f, n_traj=5):
            z = sub["z_ch"][:]
            if len(z) > 1:
                viols.append(float(np.mean(np.diff(z) < -1e-9)))
    med_viol = float(np.median(viols)) if viols else 0.0
    assert med_viol < 0.05, f"z 单调违例率中位 {med_viol:.3f} 应 < 0.05"


def test_failed_channel_truncated_to_eol():
    """失效通道: rul 末端=0, 无 EOL 后零标签窗 (P0-1 铁律根除泄漏)。"""
    feature_path, _ = _load_channels()
    seen = False
    with h5py.File(feature_path, "r") as f:
        for _key, sub_key, sub in _iter_subs(f, n_traj=10):
            if bool(sub.attrs["event_observed"]):
                seen = True
                rul = sub["rul_ch"][:]
                assert abs(rul[-1]) < 1e-6, f"{_key}/{sub_key} 失效通道末端 rul 应=0, 实={rul[-1]}"
    assert seen, "应至少一个失效通道"


def test_censored_rul_capped():
    """删失通道 v2: rul = (T−1−t)/H (H=全局任务视界)。

    F1 收口: 不再按 cap_ratio*T 封顶; 删失保留下界语义, 分母是 H 不是观测终点 T−1。
    rul 值域天然 ≤ (T−1)/H < 1 (不提供反推仿真截止时刻的能力面)。
    旧语义注释由 channel_label_v2 承接 (见 test_channel_label_v2_censored_lb_scale_by_h)。
    """
    feature_path, cfg = _load_channels()
    from src.sim.build_channel_hi import compute_mission_horizon_windows
    H = compute_mission_horizon_windows(
        duration_years=float(cfg["sim"]["duration_years"]),
        sample_period_s=float(cfg["sim"]["sample_period_s"]))
    seen = False
    with h5py.File(feature_path, "r") as f:
        for _key, sub_key, sub in _iter_subs(f, n_traj=10):
            if not bool(sub.attrs["event_observed"]):
                seen = True
                rul = sub["rul_ch"][:]
                T = len(rul)
                expect_max = (T - 1.0) / H
                assert rul.max() <= expect_max + 1e-6, \
                    f"{_key}/{sub_key} 删失 rul max={rul.max()} 应 ≤ (T-1)/H={expect_max}"
                assert rul.max() < 1.0, "H 归一后删失 rul 应 < 1"
    assert seen, "应至少一个删失通道"


def test_split_by_trajectory():
    """同 traj_id 的 16 sub_id 必须一致 (划分严格按轨迹, 不打散)。"""
    feature_path, _ = _load_channels()
    with h5py.File(feature_path, "r") as f:
        for key in sorted(f.keys())[:3]:
            traj = f[key]
            ids = {int(traj[sk].attrs["traj_id"])
                   for sk in traj.keys() if sk.startswith("sub_")}
            assert len(ids) == 1, f"{key} 子阵 traj_id 不一致: {ids}"
            # sub_id 覆盖 0..15
            sids = {int(traj[sk].attrs["sub_id"])
                    for sk in traj.keys() if sk.startswith("sub_")}
            assert sids == set(range(16)), f"{key} sub_id 应为 0..15, 实={sorted(sids)}"


def test_delta_only_from_config():
    """δ 阈值只来自 config (标定后写死, 不随实验数据变; 防阈值泄漏)。"""
    cfg = load_config(PA_CONFIG)
    deltas_cfg = cfg["channel_level"]["delta_thresholds"]
    feature_path, _ = _load_channels()
    with h5py.File(feature_path, "r") as f:
        raw = str(f.attrs.get("delta_thresholds", ""))
    stored = {}
    for kv in raw.split(","):
        if "=" in kv:
            k, v = kv.split("=")
            stored[k.strip()] = float(v)
    for k in ["R_DS", "I_DSS", "g_m", "P_out"]:
        assert k in stored, f"h5 缺 δ {k}"
        assert abs(stored[k] - float(deltas_cfg[k])) < 1e-9, \
            f"δ {k}: h5={stored[k]} vs config={deltas_cfg[k]} 不一致"


# ================================================================ canonical x 温度单位 (清零重审)

def test_build_canonical_x_t_dev_in_celsius():
    """T_dev_C = sim Tj (Kelvin) − 273.15: 值域应在器件工作温区 (°C), 而非 ~300 K 量级。

    清零重审 P0: 旧版直取 Kelvin 当 T_dev_C 用, 与源域 T_case_C (°C) 单位错位 273.15。
    """
    from src.sim.build_channel_hi import (
        build_canonical_x, SA_COL_IDSS, SA_COL_POWER, SA_COL_TJ, SA_COL_AMP)
    T = 12
    sa = np.zeros((T, 8), dtype=np.float32)
    sa[:, SA_COL_POWER] = np.linspace(1.0, 0.98, T)      # 轻微功率退化
    sa[:, SA_COL_IDSS] = np.linspace(1.0, 0.95, T)
    sa[:, SA_COL_TJ] = 333.15                            # 60 °C, sim 原生 Kelvin
    sa[:, SA_COL_AMP] = 1.0
    x = build_canonical_x(sa, duty=0.5, deltas={"I_DSS": 0.2, "P_out": 0.2})
    assert np.allclose(x[:, 1], 60.0, atol=1e-4), "T_dev_C 应为 °C (Tj_K − 273.15)"
    assert x[:, 1].max() < 200.0, "T_dev_C 出现 Kelvin 量级回归 (>200)"


# ================================================================ channel_label_v2 rul 口径 (F1 收口)

def test_mission_horizon_derived_from_sim_period():
    """H = floor(duration_years × 365.25×24×3600 / sample_period_s)，非硬编码 4088。

    F1 拍板: RUL 统一除以任务级物理视界 H（由仿真器实际时间派生），不是 0.35×视界、
    也不是 per-轨迹 EOL 归一。当前 config → floor(8×31557600/21600) = 11688。
    """
    from src.sim.build_channel_hi import compute_mission_horizon_windows

    cfg = load_config(PA_CONFIG)
    sim = cfg["sim"]
    H = compute_mission_horizon_windows(
        duration_years=float(sim["duration_years"]),
        sample_period_s=float(sim["sample_period_s"]))
    windows_per_day = 24 * 3600 / float(sim["sample_period_s"])   # 4
    expect = int(float(sim["duration_years"]) * 365.25 * windows_per_day)
    assert H == expect, f"H={H} 应为 {expect}"
    assert H == 11688, f"H 应为 11688 (8yr × 365.25d × 4窗/日), 实={H}"
    assert H != 4088, "不得回归为旧 rul_max_norm=4088"


def test_channel_label_v2_failed_rul_absolute_no_cap():
    """失效通道 v2: rul = (EOL − t) / H，不按 cap_ratio*T 截顶，末端仍=0。

    F1 拍板: 不截顶; 绝对值语义除以全局任务视界 H, 还原为绝对窗口数可逆。
    """
    from src.sim.build_channel_hi import build_channel_labels_v2, compute_mission_horizon_windows

    cfg = load_config(PA_CONFIG)
    sim = cfg["sim"]
    H = compute_mission_horizon_windows(
        duration_years=float(sim["duration_years"]),
        sample_period_s=float(sim["sample_period_s"]))

    T = 2000
    f_s = np.linspace(0.0, 1.0, T)                # dR/δR = f_s/0.35: 在 t≈0.35T 处越限, eol 足够大
    zoo = np.zeros(T)
    params = {"Delta_R": 2.0, "decay_I": 0.5, "decay_g": 0.4, "duty": 0.5, "Tj_base_C": 110.0}
    deltas = dict(cfg["channel_level"]["delta_thresholds"])
    z, hi, rul, event, eol, keep = build_channel_labels_v2(
        f_s, params, deltas, H=H, cap_ratio=None)

    assert event is True
    assert z[eol] >= 1.0 - 1e-9, "EOL 处损伤应越限"
    assert z[eol + 1] >= z[eol], "z 单调不可逆"
    assert abs(rul[eol]) < 1e-6, "失效末端 rul 应=0"
    # 绝对值可还原 (不截顶的实质断言): rul*H == eol-t 逐点
    assert np.allclose(rul[:eol] * H, (eol - np.arange(eol)).astype(float), atol=1e-3)
    assert np.all(np.diff(rul[:eol]) < 0), "失效通道 rul 应严格单调递减 (无截顶平台)"
    # 若 eol 足够晚, rul[0] 应等于 eol/H (不是被 0.35 压平)
    assert abs(rul[0] - eol / H) < 1e-6, f"rul[0] 应=eol/H 无截顶, 实={rul[0]} vs {eol/H}"


def test_channel_label_v2_censored_lb_scale_by_h():
    """删失通道 v2: rul_lb = (T−1−t) / H（观测终点下界），分母是 H 不是 T−1。

    F1 拍板: 删失保留绝对下界语义; 不允许把观测终点 T−1 当归一化分母。
    """
    from src.sim.build_channel_hi import build_channel_labels_v2, compute_mission_horizon_windows

    cfg = load_config(PA_CONFIG)
    sim = cfg["sim"]
    H = compute_mission_horizon_windows(
        duration_years=float(sim["duration_years"]),
        sample_period_s=float(sim["sample_period_s"]))

    T = 1200
    f_s = np.linspace(0.0, 0.25, T)               # 未达失效阈值 → 删失 (dR/dI/dg 均 < δ, z<1)
    params = {"Delta_R": 2.0, "decay_I": 0.5, "decay_g": 0.4, "duty": 0.5, "Tj_base_C": 110.0}
    deltas = dict(cfg["channel_level"]["delta_thresholds"])
    z, hi, rul, event, eol, keep = build_channel_labels_v2(
        f_s, params, deltas, H=H, cap_ratio=None)

    assert event is False
    assert keep == T
    assert abs(rul[0] - (T - 1) / H) < 1e-6, f"删失 rul[0] 应为 (T-1)/H, 实={rul[0]}"
    assert abs(rul[-1]) < 1e-6, "删失末端 rul 应≈0 (观测终点下界=0)"
    # 不得以 T−1 为分母（那样 rul[0] 会=1）
    assert rul[0] < 0.5, f"T−1 作分母会令 rul[0]=1; 正确除以 H 应 <1, 实={rul[0]}"


def test_h5_attrs_record_rul_scale_channel_label_v2():
    """h5 顶层 attrs 记录 rul_label_schema=channel_label_v2、rul_scale_windows、rul_capped=false。

    消费者不再自行计算尺度; bundle 与指标统一从 h5 读。
    """
    feature_path, cfg = _load_channels()
    # 数据须为 v2 口径才测; 旧 v1 h5 数据 (channel_label_schema=channel_label_v1) skip
    with h5py.File(feature_path, "r") as f:
        if str(f.attrs.get("channel_label_schema", "")) != "channel_label_v2":
            pytest.skip("h5 数据并非 channel_label_v2 (旧 v1 产物); 先重建 build_channel_hi")
    from src.sim.build_channel_hi import compute_mission_horizon_windows
    H_from_cfg = compute_mission_horizon_windows(
        duration_years=float(cfg["sim"]["duration_years"]),
        sample_period_s=float(cfg["sim"]["sample_period_s"]))
    with h5py.File(feature_path, "r") as f:
        assert str(f.attrs["channel_label_schema"]) == "channel_label_v2"
        assert str(f.attrs["rul_capped"]) == "false"
        assert float(f.attrs["rul_scale_windows"]) == float(H_from_cfg)
        assert float(f.attrs["sample_period_s"]) == float(cfg["sim"]["sample_period_s"])
        assert float(f.attrs["mission_horizon_windows"]) == float(H_from_cfg)
