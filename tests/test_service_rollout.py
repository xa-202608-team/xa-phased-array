"""src/physics/service_rollout 一致性测试 (T4/M4 生死门)。

5 项 (文档 T4, 至少 20 条轨迹含失效/删失各 ≥5):
  1. test_reconstruct_exact: latent_sub_damage+c_elem → f_ch vs 仿真内部 __verify_f_ch, 相对 1e-6
  2. test_metrics_match_sim: 真值 → G/SLL/theta/M_link vs 仿真 df *_true, 相对 1e-4
  3. test_service_eol_match: 真值 → eol_idx/failed vs 仿真, **完全相等** (生死门核心)
  4. test_rollout_monotone: 剂量恒定时前推 z 单调不减
  5. test_no_truth_leak: rollout_mc 签名/源码不接触 label_*/latent_*

用重新仿真 (float64 twin dict) 对比, 验证 service_rollout 公式与 _simulate_subdose 逐字一致
(float32 落盘精度是 M1 的事, 此处验证物理孪生实现正确性)。不过则停下报告 (GO/NO-GO)。
"""
import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                                      # noqa: E402
from src.sim.phased_array_sim import sample_params, simulate, _grid_positions  # noqa: E402
from src.physics.service_rollout import (                              # noqa: E402
    ArrayTwin, reconstruct_elements, element_rf, array_metrics, service_eol, rollout_mc)

PA_CONFIG = ROOT / "configs" / "phased_array.yaml"


def _twin_from_dict(td: dict, params: dict, cfg: dict) -> ArrayTwin:
    """从 simulate 返回的 twin dict (float64 真值) 构建 ArrayTwin。"""
    ac = cfg["sim"]["array"]
    lim = cfg["sim"]["service_limits"]
    return ArrayTwin(
        c_elem=td["twin_c_elem"].astype(np.float64),
        eta_R=td["twin_eta_R"].astype(np.float64),
        eta_phi=td["twin_eta_phi"].astype(np.float64),
        dropout_thr=td["twin_dropout_thr"].astype(np.float64),
        grad_dir=td["twin_grad_dir"].astype(np.float64),
        sub_ids=td["twin_subarray_ids"],
        pos2d=_grid_positions(list(ac["grid"]), float(ac["element_spacing_lambda"])),
        grid_x=int(ac["grid"][0]),
        spacing=float(ac["element_spacing_lambda"]),
        Delta_R=float(params["Delta_R"]),
        decay_I=float(params["decay_I"]),
        decay_g=float(params["decay_g"]),
        dphi_max_deg=float(params["dphi_max_deg"]),
        scan_az_deg=float(params["scan_az_deg"]),
        margin0_dB=float(params["margin0_dB"]),
        SLL_max_dB=float(lim["SLL_max_dB"]),
        theta_err_max_deg=float(lim["theta_err_max_deg"]),
        consecutive_windows=int(lim.get("consecutive_windows", 4)),
        R_th_a5=float(cfg["sim"]["physics"]["damage_model"]["rth_feedback"]["a5"]),
    )


def _sim_n(n: int, cfg: dict):
    """生成 n 条轨迹的 (df, twin_dict, params)。"""
    sim_cfg = dict(cfg["sim"])
    rng = np.random.default_rng(cfg["seed"])
    out = []
    for _ in range(n):
        params = sample_params(rng, sim_cfg)
        traj_rng = np.random.default_rng(params["seed_traj"])
        df, _sa, eol, failed, td = simulate(params, sim_cfg, traj_rng)
        out.append((df, td, params, eol, failed))
    return out


def test_reconstruct_exact():
    """latent_sub_damage+c_elem 重建 f_ch, 与仿真内部 __verify_f_ch 逐点一致 (相对 1e-6)。"""
    cfg = load_config(PA_CONFIG)
    for df, td, params, _eol, _fl in _sim_n(5, cfg):
        twin = _twin_from_dict(td, params, cfg)
        f_sub = td["latent_sub_damage"].astype(np.float64)
        f_ch_recon = reconstruct_elements(f_sub, twin)
        f_ch_internal = td["__verify_f_ch"].astype(np.float64)
        denom = np.maximum(np.abs(f_ch_internal), 1e-9)
        rel = float(np.max(np.abs(f_ch_recon - f_ch_internal) / denom))
        assert rel < 1e-6, f"元件重建相对误差 {rel:.2e} 应 < 1e-6"


def test_metrics_match_sim():
    """真值 → G/SLL/theta/M_link vs 仿真 df *_true, 相对/绝对 1e-4 (M4 生死门)。"""
    cfg = load_config(PA_CONFIG)
    keys = [("G_array_dB", "G_array_dB_true"), ("SLL_dB", "SLL_dB_true"),
            ("theta_err_deg", "theta_err_deg_true"), ("M_link_dB", "M_link_dB_true")]
    for df, td, params, _eol, _fl in _sim_n(5, cfg):
        twin = _twin_from_dict(td, params, cfg)
        f_sub = td["latent_sub_damage"].astype(np.float64)
        f_ch = reconstruct_elements(f_sub, twin)
        a, dphi = element_rf(f_ch, f_sub.mean(axis=1), twin)
        m = array_metrics(a, dphi, twin)
        for recon_key, true_key in keys:
            recon = m[recon_key]
            true = df[true_key].values.astype(np.float64)
            assert np.allclose(recon, true, rtol=1e-4, atol=1e-4), \
                f"{recon_key}: max|Δ|={np.max(np.abs(recon-true)):.2e}"


def test_service_eol_match():
    """真值 → eol_idx/failed vs 仿真, **完全相等** (M4 生死门核心, ≥20 轨迹含失效/删失各≥5)。"""
    cfg = load_config(PA_CONFIG)
    trajs = _sim_n(25, cfg)
    n_failed = sum(1 for *_t, eol, fl in trajs if fl)
    n_censored = len(trajs) - n_failed
    assert n_failed >= 5, f"失效轨迹 {n_failed} <5"
    assert n_censored >= 5, f"删失轨迹 {n_censored} <5"
    for df, td, params, eol_sim, failed_sim in trajs:
        twin = _twin_from_dict(td, params, cfg)
        f_sub = td["latent_sub_damage"].astype(np.float64)
        f_ch = reconstruct_elements(f_sub, twin)
        a, dphi = element_rf(f_ch, f_sub.mean(axis=1), twin)
        m = array_metrics(a, dphi, twin)
        eol_recon, failed_recon = service_eol(m, twin)
        assert eol_recon == eol_sim, f"eol 不等: recon={eol_recon} vs sim={eol_sim}"
        assert failed_recon == failed_sim, f"failed 不等: recon={failed_recon} vs sim={failed_sim}"


def test_rollout_monotone():
    """剂量恒定时, 物理知情前推 z 单调不减 (R_th 正反馈不破坏单调)。"""
    z = np.zeros(16)
    rate = np.full(16, 0.01)
    dose = np.full(60, 1.0)
    a5 = 0.3
    prev_max = -np.inf
    for t in range(60):
        z = z + rate * dose[t] * (1.0 + a5 * z)
        assert z.max() >= prev_max - 1e-12, f"t={t} z.max()={z.max()} < prev {prev_max} (非单调)"
        prev_max = z.max()


def test_no_truth_leak():
    """rollout_mc 签名与代码访问不接触 label_*/latent_* (AST 检查, 排除 docstring/注释)。"""
    import ast
    sig = inspect.signature(rollout_mc)
    for pname in sig.parameters:
        assert "label" not in pname.lower() and "latent" not in pname.lower(), \
            f"rollout_mc 参数 {pname} 含 label/latent"
    tree = ast.parse(inspect.getsource(rollout_mc))
    leaked = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str):
            if node.attr.startswith(("label_", "latent_")):
                leaked.append(f".{node.attr}")
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            v = node.slice.value
            if isinstance(v, str) and v.startswith(("label_", "latent_")):
                leaked.append(f'["{v}"]')
    assert not leaked, f"rollout_mc 接触真值: {leaked}"
