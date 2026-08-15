"""相控阵仿真物理合理性 + 应力敏感性测试 (PA3 重做验收, G1 物理敏感性门)。

覆盖:
  - 潜在 damage 严格非减 (Arrhenius 累积必然)
  - 去噪 EIRP 与 damage 强负相关 (原始观测允许波动, GPT §五 单调性口径)
  - M_link_true 随时间下降
  - 失效在退化中后期 (非起点瞬时)
  - 阵列级列 + 子阵块 (T,16,8) 结构完整
  - G1 门: Ea↑/Tj↑/duty↑ 加重损伤 (Arrhenius 应力敏感性, 修复 P0-2)
  - 低应力轨迹可 survive (右删失, 修复 P0-2 自然分化)
  - 固定种子两次仿真哈希一致
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                       # noqa: E402
from src.sim.phased_array_sim import sample_params, simulate      # noqa: E402

PA_CONFIG = ROOT / "configs" / "phased_array.yaml"


def _short_sim_cfg(cfg):
    s = dict(cfg["sim"])
    s["duration_years"] = 1.0
    s["sample_period_s"] = 43200.0
    s["physics_dt_s"] = 900.0
    return s


def _make_traj(seed):
    cfg = load_config(PA_CONFIG)
    set_seed(seed, cfg["reproducibility"]["deterministic"])
    rng = np.random.default_rng(seed)
    sc = _short_sim_cfg(cfg)
    params = sample_params(rng, sc)
    traj_rng = np.random.default_rng(params["seed_traj"])
    df, sa, eol, failed, _ = simulate(params, sc, traj_rng)
    return df, sa, params, failed, eol


def _damage_end(params, sc):
    tr = np.random.default_rng(params["seed_traj"])
    df, _, _, _, _ = simulate(params, sc, tr)
    return float(df["damage"].iloc[-1])


def test_damage_strictly_nondecreasing():
    """潜在 damage 严格非减 (Arrhenius 累积, 物理必然)。"""
    df, _, _, _, _ = _make_traj(1)
    diffs = np.diff(df["damage"].values)
    assert np.all(diffs >= -1e-12), "damage 应严格非减"


def test_eirp_correlates_with_damage():
    """去噪后 EIRP_norm 与 damage 强负相关 (原始观测允许短期波动)。"""
    df, _, _, _, _ = _make_traj(2)
    eirp = pd.Series(df["EIRP_norm"].values).rolling(30, 1).median().values
    dmg = pd.Series(df["damage"].values).rolling(30, 1).median().values
    corr = np.corrcoef(eirp, dmg)[0, 1]
    assert corr < -0.7, f"去噪 EIRP-damage 相关 {corr:.3f} 应 < -0.7"


def test_link_margin_decreasing():
    """M_link_true 应随时间下降 (链路余量耗尽)。"""
    df, _, _, _, _ = _make_traj(3)
    ml = pd.Series(df["M_link_dB_true"].values).rolling(30, 1).mean().values
    assert ml[-1] < ml[0], f"M_link 末端 {ml[-1]:.2f} 应低于初值 {ml[0]:.2f}"


def test_failure_not_at_start():
    """失效应在退化中后期 (非起点瞬时)。"""
    for s in range(8):
        df, _, _, failed, eol = _make_traj(s)
        if failed:
            assert eol > 5, f"seed {s} EOL={eol} 不应在起点"
            return
    assert False, "8 条种子应至少一条失效"


def test_output_columns_and_subarray():
    """阵列级列 + 子阵块结构完整 (修复 P0-4 数据接口)。"""
    df, sa, _, _, _ = _make_traj(4)
    required = {"damage", "EIRP_norm", "M_link_dB_true", "SLL_dB_true",
                "theta_err_deg_true", "label_fail"}
    assert required.issubset(set(df.columns)), required - set(df.columns)
    assert sa.ndim == 3 and sa.shape[1] == 16 and sa.shape[2] == 8, \
        f"子阵块 shape {sa.shape} 应为 (T,16,8)"


def test_stress_sensitivity_arrhenius():
    """G1 物理敏感性门: Ea↑/Tj↑/duty↑ 应加重损伤 (修复 P0-2 应力失效)。"""
    cfg = load_config(PA_CONFIG)
    sc = _short_sim_cfg(cfg)
    rng = np.random.default_rng(99)
    base = sample_params(rng, sc)
    d_lo_Ea = _damage_end({**base, "Ea_eV": 0.7}, sc)
    d_hi_Ea = _damage_end({**base, "Ea_eV": 1.1}, sc)
    assert d_hi_Ea > d_lo_Ea * 1.5, f"Ea 敏感性失效: {d_lo_Ea:.3f} vs {d_hi_Ea:.3f}"
    d_hot = _damage_end({**base, "Tj_base_C": base["Tj_base_C"] + 30}, sc)
    d_cold = _damage_end({**base, "Tj_base_C": base["Tj_base_C"] - 30}, sc)
    assert d_hot > d_cold * 2.0, f"Tj 敏感性失效: hot={d_hot:.3f} cold={d_cold:.3f}"
    d_hi_duty = _damage_end({**base, "duty": 0.8}, sc)
    d_lo_duty = _damage_end({**base, "duty": 0.3}, sc)
    assert d_hi_duty > d_lo_duty * 1.3, f"duty 敏感性失效: {d_lo_duty:.3f} vs {d_hi_duty:.3f}"


def test_low_stress_can_survive():
    """低应力轨迹应能 survive (右删失, 修复 P0-2 自然分化)。"""
    cfg = load_config(PA_CONFIG)
    sc = _short_sim_cfg(cfg)
    survive_seen = False
    for s in range(12):
        rng = np.random.default_rng(500 + s)
        params = sample_params(rng, sc)
        params["Ea_eV"] = 0.7
        params["Tj_base_C"] = 95.0
        params["duty"] = 0.3
        df, _, _, failed, _ = simulate(
            params, sc, np.random.default_rng(params["seed_traj"]))
        if not failed:
            survive_seen = True
            break
    assert survive_seen, "低应力轨迹应能 survive (体现优雅降级/右删失)"


def test_seed_reproducibility():
    """同一 seed 两次仿真输出哈希一致。"""
    df1, _, _, _, _ = _make_traj(7)
    df2, _, _, _, _ = _make_traj(7)
    h1 = pd.util.hash_pandas_object(df1, index=True).values.tobytes()
    h2 = pd.util.hash_pandas_object(df2, index=True).values.tobytes()
    assert h1 == h2, "同 seed 两次仿真不一致"


def test_subarray_dose_dispersion():
    """T1/M1: 子阵级独立损伤积分 → 子阵间末端损伤有动力学分散 (非互为缩放)。

    旧标量路径下 16 子阵共享 f(t), 末端 CV≈0 (仅静态 boost 差异);
    subdose 路径下每子阵独立 life_scale_s, CV 中位应 > 0.15 (文档 M1 门)。
    """
    cfg = load_config(PA_CONFIG)
    sc = _short_sim_cfg(cfg)
    cvs = []
    for s in range(30):
        rng = np.random.default_rng(1000 + s)
        params = sample_params(rng, sc)
        traj_rng = np.random.default_rng(params["seed_traj"])
        _, _, _, _, twin = simulate(params, sc, traj_rng)
        assert twin is not None, "subdose 路径应返回 twin"
        f_end = twin["latent_sub_damage"][-1]            # (16,) 末端子阵损伤
        cvs.append(np.std(f_end) / (abs(np.mean(f_end)) + 1e-9))
    med_cv = float(np.median(cvs))
    assert med_cv > 0.15, f"子阵末端损伤 CV 中位 {med_cv:.3f} 应 > 0.15 (独立动力学)"


def test_element_reconstruction_exact():
    """T1/M1: latent_sub_damage + twin_c_elem 重建 f_ch, 与仿真内部逐点一致 (M4 前置)。

    重建公式: f_ch[:, i] = latent_sub_damage[:, s(i)] * twin_c_elem[i] (静态缩放, 与 t 无关)。
    float32 落盘精度下相对误差须 < 1e-6 (M4 物理孪生一致性前提)。
    """
    cfg = load_config(PA_CONFIG)
    sc = _short_sim_cfg(cfg)
    rng = np.random.default_rng(42)
    params = sample_params(rng, sc)
    traj_rng = np.random.default_rng(params["seed_traj"])
    df, _, _, _, twin = simulate(params, sc, traj_rng)
    assert twin is not None
    f_sub = twin["latent_sub_damage"].astype(np.float64)     # (T, 16)
    c_elem = twin["twin_c_elem"].astype(np.float64)          # (256,)
    sa_ids = twin["twin_subarray_ids"]                       # (256,)
    f_ch_internal = twin["__verify_f_ch"].astype(np.float64) # (T, 256) 仿真内部 (float32)
    f_ch_recon = np.clip(f_sub[:, sa_ids] * c_elem[None, :], 0.0, None)
    denom = np.maximum(np.abs(f_ch_internal), 1e-9)
    rel_err = float(np.max(np.abs(f_ch_recon - f_ch_internal) / denom))
    assert rel_err < 1e-6, f"元件重建相对误差 {rel_err:.2e} 应 < 1e-6 (M4 前置)"
