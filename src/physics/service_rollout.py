"""src/physics/service_rollout.py

确定性物理孪生 (T4/M4 生死门): 由子阵损伤真值/预测前推阵列服务性能与寿命分布。

一致性链 (与 src/sim/phased_array_sim._simulate_subdose 逐字一致, M4 生死门):
  reconstruct_elements: f_sub(T,16) + c_elem → f_ch(T,256)
  element_rf:           f_ch + twin → (a, dphi_rad)  [RDS/IDSS/gm/P_ratio/dphi_grad/dropout]
  array_metrics:        a, dphi → {G/SLL/theta_err/M_link}  [复用 _array_pattern/_refine_peak]
  service_eol:          metrics → (eol_idx, failed)  [violate + consecutive 判据]

前推 (T8 服务层评估用):
  rollout_mc: z_hat + dose → 服务寿命分布 (物理知情外推 + MC)

关键纪律: 本模块只接触 twin_* 量与模型预测, 绝不读 label_*/latent_* (test_no_truth_leak)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArrayTwin:
    """一条轨迹的确定性孪生参数 (全部 twin_only, 不进模型输入 x)。"""
    c_elem: np.ndarray        # (256,) 元件相对子阵静态缩放
    eta_R: np.ndarray         # (256,) 幅度制造散布
    eta_phi: np.ndarray       # (256,) 相位随机系数
    dropout_thr: np.ndarray   # (256,) Weibull 退出阈值 (含聚簇修正)
    grad_dir: np.ndarray      # (2,) 相位空间梯度方向
    sub_ids: np.ndarray       # (256,) 元件→子阵映射
    pos2d: np.ndarray         # (256,2) 元件位置
    grid_x: int               # 阵列 x 元件数 (du=1/(grid_x*spacing))
    spacing: float            # 元件间距 (波长单位)
    Delta_R: float
    decay_I: float
    decay_g: float
    dphi_max_deg: float
    scan_az_deg: float
    margin0_dB: float
    SLL_max_dB: float
    theta_err_max_deg: float
    consecutive_windows: int
    R_th_a5: float            # R_th 反馈系数 (rollout_mc 前推用)

    @classmethod
    def from_h5_group(cls, g, service_limits: dict | None = None,
                      array_cfg: dict | None = None, rth_a5: float = 0.3) -> "ArrayTwin":
        """从 h5 轨迹组构建孪生 (sim_v2: twin 在 g 直接; channel_features: 在 g['twin/'])。"""
        twin_grp = g["twin"] if "twin" in g else g
        def _get(name):
            src = twin_grp if name in twin_grp else g
            return src[name][:]

        grid = list(array_cfg["grid"]) if array_cfg else [16, 16]
        d_lambda = float(array_cfg["element_spacing_lambda"]) if array_cfg else 0.5
        from src.sim.phased_array_sim import _grid_positions
        pos2d = _grid_positions(grid, d_lambda)
        lim = service_limits or {}
        return cls(
            c_elem=_get("twin_c_elem").astype(np.float64),
            eta_R=_get("twin_eta_R").astype(np.float64),
            eta_phi=_get("twin_eta_phi").astype(np.float64),
            dropout_thr=_get("twin_dropout_thr").astype(np.float64),
            grad_dir=_get("twin_grad_dir").astype(np.float64),
            sub_ids=_get("twin_subarray_ids"),
            pos2d=pos2d,
            grid_x=int(grid[0]),
            spacing=d_lambda,
            Delta_R=float(g.attrs["Delta_R"]),
            decay_I=float(g.attrs["decay_I"]),
            decay_g=float(g.attrs["decay_g"]),
            dphi_max_deg=float(g.attrs["dphi_max_deg"]),
            scan_az_deg=float(g.attrs["scan_az_deg"]),
            margin0_dB=float(g.attrs["margin0_dB"]),
            SLL_max_dB=float(lim.get("SLL_max_dB", -8.0)),
            theta_err_max_deg=float(lim.get("theta_err_max_deg", 0.5)),
            consecutive_windows=int(lim.get("consecutive_windows", 4)),
            R_th_a5=float(rth_a5),
        )


def reconstruct_elements(f_sub: np.ndarray, twin: ArrayTwin) -> np.ndarray:
    """(T,16) 子阵损伤 → (T,256) 元件损伤。f_ch[:,i] = f_sub[:,s(i)] · c_elem[i]。

    与 _simulate_subdose 定义逐字一致 (test_reconstruct_exact 验证 1e-6)。
    """
    return np.clip(f_sub[:, twin.sub_ids] * twin.c_elem[None, :], 0.0, None)


def element_rf(f_ch: np.ndarray, f_track: np.ndarray, twin: ArrayTwin):
    """(T,256) 元件损伤 → (a (T,256), dphi_rad (T,256))。

    与 _simulate_subdose 第二级逐字一致; f_track=轨迹级标量 (dphi_grad 用, =f_sub.mean)。
    """
    RDS = 1.0 + (twin.Delta_R - 1.0) * f_ch
    IDSS = np.clip(1.0 - twin.decay_I * f_ch, 0.1, 1.0)
    gm = np.clip(1.0 - twin.decay_g * f_ch, 0.1, 1.0)
    P_ratio = IDSS * gm / RDS
    a = np.sqrt(np.clip(P_ratio, 0.0, None)) * twin.eta_R[None, :]
    grad_amp = np.deg2rad(twin.dphi_max_deg) * 0.3
    dphi_grad = grad_amp * np.outer(
        f_track, twin.grad_dir[0] * twin.pos2d[:, 0] + twin.grad_dir[1] * twin.pos2d[:, 1])
    dphi_rad = np.deg2rad(twin.dphi_max_deg) * f_ch * twin.eta_phi[None, :] + dphi_grad
    dropout_mask = f_ch > twin.dropout_thr[None, :]
    a = np.where(dropout_mask, 0.0, a)
    return a, dphi_rad


def array_metrics(a: np.ndarray, dphi_rad: np.ndarray, twin: ArrayTwin) -> dict:
    """→ {G_array_dB, SLL_dB, theta_err_deg, M_link_dB} 各 (T,)。

    复用 _array_pattern/_refine_peak/主瓣首零点窗, 与仿真严格同实现 (M4 一致性前提)。
    """
    from src.sim.phased_array_sim import _array_pattern, _refine_peak, THETA_COARSE_DEG
    n_out = a.shape[0]
    n_ch = a.shape[1]
    pos_x = twin.pos2d[:, 0]
    u0 = np.sin(np.deg2rad(twin.scan_az_deg))
    patt = _array_pattern(a, dphi_rad, pos_x, np.sin(np.deg2rad(THETA_COARSE_DEG)), u0)
    peak_idx = np.argmax(patt, axis=1)
    peak_pow = np.empty(n_out)
    theta_peak = np.empty(n_out)
    for ti in range(n_out):
        pw, th = _refine_peak(patt[ti], THETA_COARSE_DEG, int(peak_idx[ti]))
        peak_pow[ti] = pw
        theta_peak[ti] = th
    du = 1.0 / (twin.grid_x * twin.spacing)
    th_left = np.degrees(np.arcsin(np.clip(u0 - du, -0.99, 0.99)))
    th_right = np.degrees(np.arcsin(np.clip(u0 + du, -0.99, 0.99)))
    main_mask = (THETA_COARSE_DEG >= th_left) & (THETA_COARSE_DEG <= th_right)
    patt_sl = patt.copy()
    patt_sl[:, main_mask] = 0.0
    sll_lin = np.max(patt_sl, axis=1) / (peak_pow + 1e-30)
    SLL = np.clip(10.0 * np.log10(sll_lin + 1e-30), -50.0, 0.0)
    G = 10.0 * np.log10(peak_pow / (n_ch * n_ch) + 1e-30)
    dEIRP = np.clip(G[0] - G, 0.0, 40.0)
    M_link = twin.margin0_dB - dEIRP
    theta_err = theta_peak - twin.scan_az_deg
    return {
        "G_array_dB": G,
        "SLL_dB": SLL,
        "theta_err_deg": theta_err,
        "M_link_dB": M_link,
    }


def service_eol(metrics: dict, twin: ArrayTwin):
    """→ (eol_idx, failed)。持续 N_c 窗越限判据, 与仿真严格同实现。"""
    violate = ((metrics["M_link_dB"] <= 0.0)
               | (metrics["SLL_dB"] > twin.SLL_max_dB)
               | (np.abs(metrics["theta_err_deg"]) > twin.theta_err_max_deg))
    n_out = len(violate)
    consec = twin.consecutive_windows
    eol_idx = n_out - 1
    failed = False
    if consec <= 1:
        if violate.any():
            eol_idx = int(np.argmax(violate))
            failed = True
    else:
        cs = np.convolve(violate.astype(int), np.ones(consec, dtype=int), mode="full")[:n_out]
        if (cs >= consec).any():
            eol_idx = int(np.argmax(cs >= consec))
            failed = True
    return eol_idx, failed


def rollout_mc(z_hat: np.ndarray, rate_hat: dict, dose: np.ndarray, twin: ArrayTwin,
               horizon: int, n_mc: int = 200, common_mode_frac: float = 0.6,
               rng: np.random.Generator | None = None) -> dict:
    """由模型预测的子阵损伤前推服务寿命分布 (物理知情外推 + MC)。

    z_hat: (16,) 当前子阵损伤中位; rate_hat: {p10/p50/p90: (16,)} 损伤速率分位
    dose: (horizon,) 归一化剂量率 (由遥测 Tj/duty 按 Arrhenius 算, 标称 Ea, 非真值)
    前推律: ẑ_s(t+1) = ẑ_s(t) + rate_s · dose[t] · (1 + a5·ẑ_s(t))  (R_th 正反馈)
    MC: rate_s^(m) = rate_p50 · exp(σ_com·ξ^m + σ_s·ξ_s^m); σ 由 P10/P90 反解 (对数正态)

    只用 z_hat/rate_hat/dose/twin, 绝不读 label_*/latent_* (test_no_truth_leak)。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    rate50 = np.asarray(rate_hat["p50"], dtype=np.float64)
    rate10 = np.asarray(rate_hat["p10"], dtype=np.float64)
    rate90 = np.asarray(rate_hat["p90"], dtype=np.float64)
    # σ 从 P10/P90 反解 (对数正态: ln(p90/p50)=σ·Φ⁻¹(0.9)≈1.2816σ)
    Z90 = 1.2816
    sigma_total = np.maximum(
        (np.log(np.maximum(rate90, 1e-12) / np.maximum(rate50, 1e-12))
         + np.log(np.maximum(rate50, 1e-12) / np.maximum(rate10, 1e-12))) / (2 * Z90), 1e-4)
    sigma_com = np.sqrt(sigma_total ** 2 * common_mode_frac)
    sigma_sub = np.sqrt(sigma_total ** 2 * (1.0 - common_mode_frac))
    a5 = twin.R_th_a5
    horizon = min(int(horizon), len(dose))
    eols = np.empty(n_mc, dtype=np.int64)
    z_init = np.asarray(z_hat, dtype=np.float64)
    for m in range(n_mc):
        xi_com = rng.standard_normal()
        rate_m = rate50 * np.exp(sigma_com * xi_com + sigma_sub * rng.standard_normal(16))
        z = z_init.copy()
        eol = horizon
        for t in range(horizon):
            z = z + rate_m * dose[t] * (1.0 + a5 * z)
            if (z >= 1.0).any():
                eol = t
                break
        eols[m] = eol
    return {
        "eol_p10": int(np.percentile(eols, 10)),
        "eol_p50": int(np.percentile(eols, 50)),
        "eol_p90": int(np.percentile(eols, 90)),
        "fail_prob": float(np.mean(eols < horizon)),
        "bottleneck_sub": int(np.argmax(z_init)),
    }
