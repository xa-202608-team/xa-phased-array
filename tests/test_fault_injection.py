"""tests/test_fault_injection.py — 故障注入 §5.7 五项测试。

配对注入设计: 60 条故障轨迹是 60 条标称轨迹的克隆 (同 seed/同采样参数),
唯一差别是注入项。以下测试验证配对干净性、注入可见性、幅值确定性。
"""
import numpy as np
import pandas as pd
import pytest

from src.sim.phased_array_sim import simulate, _build_fault_cfg, sample_params
from src.utils.config import load_config
from src.utils.seed import set_seed


def _smoke_cfg():
    """小型 config 用于快速测试 (5 轨迹, 1 年, 12h 采样)。"""
    cfg = load_config("configs/phased_array.yaml")
    sim_cfg = dict(cfg["sim"])
    sim_cfg.update(n_traj=5, duration_years=1.0, sample_period_s=43200.0, physics_dt_s=900.0)
    sim_cfg.setdefault("physics", {}).setdefault("subarray_dose", {})["enabled"] = True
    return cfg, sim_cfg


def _run_one(cfg, sim_cfg, seed=42, fault_cfg=None):
    """跑一条轨迹, 返回 (df, sa, eol, failed, params)。"""
    set_seed(seed, True, False)
    rng = np.random.default_rng(seed)
    params = sample_params(rng, sim_cfg)
    traj_rng = np.random.default_rng(params["seed_traj"])
    df, sa, eol, failed, twin = simulate(params, sim_cfg, traj_rng, fault_cfg=fault_cfg)
    return df, sa, eol, failed, params


# -----------------------------------------------------------------------
def test_none_bit_exact():
    """--inject none 必须逐位复现标称轨迹 (故障参数不消耗主流 RNG)。"""
    cfg, sim_cfg = _smoke_cfg()

    df_nom, sa_nom, eol_nom, fail_nom, _ = _run_one(cfg, sim_cfg, fault_cfg=None)
    df_none, sa_none, eol_none, fail_none, _ = _run_one(cfg, sim_cfg, fault_cfg={"type": "none"})

    pd.testing.assert_frame_equal(df_nom, df_none, check_exact=True)
    np.testing.assert_array_equal(sa_nom, sa_none)
    assert eol_nom == eol_none
    assert fail_nom == fail_none


def test_pre_injection_identical():
    """注入时点之前的状态轨迹与配对标称逐位相同。"""
    cfg, sim_cfg = _smoke_cfg()

    df_nom, _, eol_nom, _, params = _run_one(cfg, sim_cfg)

    n_win = len(df_nom)
    spw = int(sim_cfg["sample_period_s"] / sim_cfg["physics_dt_s"])
    fc = _build_fault_cfg(cfg, "rth_step", traj_id=0, eol_nom=eol_nom,
                          n_win=n_win, steps_per_win=spw)

    df_fault, _, _, _, _ = _run_one(cfg, sim_cfg, fault_cfg=fc)

    pre = fc["start_win"]
    assert pre > 0, "注入时点必须 > 0"
    pd.testing.assert_frame_equal(
        df_nom.iloc[:pre].reset_index(drop=True),
        df_fault.iloc[:pre].reset_index(drop=True),
        check_exact=True,
    )


def test_amplitude_from_config_only():
    """幅值完全由 config 决定, 给定 (traj_id, fault_type) 完全确定 (不含 RNG)。"""
    cfg, sim_cfg = _smoke_cfg()
    n_win, spw = 100, 48

    # 同输入 → 同输出 (确定性)
    for ft in ["rth_step", "thermal_bias", "channel_open", "cal_freeze"]:
        fc1 = _build_fault_cfg(cfg, ft, 0, 50, n_win, spw)
        fc2 = _build_fault_cfg(cfg, ft, 0, 50, n_win, spw)
        assert fc1 == fc2, f"{ft}: 同输入应得同输出"

    # 不同 traj_id → 不同 sub_id (F1/F3)
    fc_a = _build_fault_cfg(cfg, "rth_step", 0, 50, n_win, spw)
    fc_b = _build_fault_cfg(cfg, "rth_step", 1, 50, n_win, spw)
    assert fc_a["sub_id"] != fc_b["sub_id"], "不同 traj_id 应得不同 sub_id"


def test_paired_params_identical():
    """配对轨迹的 Ea/Tj/duty/life_scale 逐位相同 (故障不改变采样参数)。"""
    cfg, sim_cfg = _smoke_cfg()
    set_seed(42, True, False)

    rng1 = np.random.default_rng(42)
    p_nom = sample_params(rng1, sim_cfg)

    rng2 = np.random.default_rng(42)
    p_fault = sample_params(rng2, sim_cfg)

    for key in ["Ea_eV", "Tj_base_C", "duty", "life_scale_years", "Delta_R", "scan_az_deg"]:
        assert p_nom[key] == p_fault[key], f"配对参数 {key} 不一致"


def test_fault_types_visible():
    """四种故障确实改变了输出 (注入后标称≠故障)。"""
    cfg, sim_cfg = _smoke_cfg()

    df_nom, _, eol_nom, _, _ = _run_one(cfg, sim_cfg)
    n_win = len(df_nom)
    spw = int(sim_cfg["sample_period_s"] / sim_cfg["physics_dt_s"])

    for ft in ["rth_step", "thermal_bias", "channel_open", "cal_freeze"]:
        fc = _build_fault_cfg(cfg, ft, traj_id=0, eol_nom=eol_nom,
                              n_win=n_win, steps_per_win=spw)
        df_f, _, eol_f, fail_f, _ = _run_one(cfg, sim_cfg, fault_cfg=fc)

        # 注入后轨迹必须有变化 (不是全等)
        post_nom = df_nom.iloc[fc["start_win"]:].values
        post_fault = df_f.iloc[fc["start_win"]:].values
        assert not np.allclose(post_nom, post_fault), \
            f"{ft}: 注入后轨迹与标称相同 (注入无效)"
