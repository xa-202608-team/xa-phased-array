"""sim/phased_array_sim.py

相控阵 T/R 退化仿真器 — 三级退化链 + 阵列方向图, 参数 YAML 化, 固定种子。
对应 docs/phased_array_dev_plan.md PA3 / 方案 v1.0 §3 §5。

二轮重做 (修复 GPT 十一/十二/十三/十四/十六):
  - Tj 6h 窗口聚合 (十一): 不取起点瞬时 (6h=4×90min 同相位), 输出 mean/max/min
  - SLL 左右首零点 (十二): 主瓣窗 = [arcsin(u0-du), arcsin(u0+du)], 非绝对角度
  - 空间相关相位梯度 (十三): dphi += a·(grad·pos2d), 随 damage 漂移形成持续指向偏移
  - dropout 入子阵功率 (十四): P_effective = where(dropout, 0, P_ratio)
  - 流式 HDF5 (十六): 逐轨迹写盘, 不全存内存 (200 轨迹 ~1.4GiB 风险)

一级 (Arrhenius 累积, life_scale 独立于 duration):
  accel=exp(-Ea/kB·(1/Tj-1/Tref)); eff_age=∫accel·duty; damage=(eff_age/life_scale)^m
二级 (GaN 参数漂移 + 空间相位梯度): R_DS↑/I_DSS↓/g_m↓/δφ(独立+梯度)
三级 (1D 方位切面 AF, 主瓣抛物线插值): EIRP/SLL/θ_err/M_link

服务寿命 (多维越限 + 持续判据): 连续 N 窗口 M_link≤0 ∨ SLL>SLL_max ∨ |θ_err|>θ_max
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed            # noqa: E402

SEC_PER_YEAR = 365.25 * 24 * 3600
K_B = 8.617e-5
TREF_K = 293.15
ORBIT_THERMAL_PERIOD_S = 90 * 60
THETA_COARSE_DEG = np.arange(-60.0, 60.01, 1.0)
DYNAMICS_ID = "leo_coupled_v1"
TARGET_CONDITION_SCHEMA = "target_ood_conditions_v1"


def _mid(a, b):
    return 0.5 * (float(a) + float(b))


def _nominal_life_ref(physics: dict, duration_years: float) -> float:
    Ea = _mid(*physics["Ea_eV_range"])
    Tj_K = _mid(*physics["Tj_range_C"]) + 273.15
    duty = _mid(*physics["duty_cycle_range"])
    accel = np.exp(-Ea / K_B * (1.0 / Tj_K - 1.0 / TREF_K))
    # P1 #2: life_ref 含 Coffin-Manson factor (中位 ΔTj), 使中位轨迹 damage_end~1 (标定到 duration 末失效)
    cm = physics.get("damage_model", {}).get("coffin_manson", {})
    if bool(cm.get("enabled", True)):
        deltaTj = 2.0 * 12.5     # 中位 Tj_amp 峰峰 (sample_params Tj_amp_range 5-20)
        cmf = (deltaTj / float(cm.get("deltaT_ref_K", 20.0))) ** float(cm.get("exponent_n", 3.0))
    else:
        cmf = 1.0
    return float(accel * duty * cmf * duration_years)


def sample_params(rng: np.random.Generator, sim_cfg: dict) -> dict:
    U = lambda a, b: float(rng.uniform(a, b))
    p = sim_cfg["physics"]
    arr = sim_cfg["array"]
    lb = sim_cfg["link_budget"]
    sigma = float(p["device_spread_sigma"])
    life_sigma = float(p.get("life_scale_sigma", 0.5))
    duration = float(sim_cfg["duration_years"])
    life_ref = float(p.get("nominal_life_effective_years", _nominal_life_ref(p, duration)))
    return dict(
        Ea_eV=U(*p["Ea_eV_range"]),
        m=float(p["time_exp_m"]),
        Tj_base_C=U(*p["Tj_range_C"]),
        Tj_amp_C=U(5.0, 20.0),
        Tj_drift_C_per_year=U(0.3, 1.0),        # 长期升温 (退化发热 + 季节, 修复 GPT 十一)
        duty=U(*p["duty_cycle_range"]),
        Delta_R=U(*p["R_DS_drift_Delta_range"]),
        decay_I=U(*p["IDSS_decay_range"]),
        decay_g=U(*p["gm_decay_range"]),
        dphi_max_deg=U(*p["phase_drift_deg_range"]),
        life_scale_years=life_ref * float(np.exp(life_sigma * rng.standard_normal())),
        scan_az_deg=U(*arr["scan_az_deg_range"]),
        margin0_dB=U(*lb["initial_margin_dB_range"]),
        weibull_beta=float(p["channel_dropout_weibull"]["beta"]),
        seed_traj=int(rng.integers(0, 2**31)),
    )


def sample_gan_target_params(rng: np.random.Generator, sim_cfg: dict) -> dict:
    """采样 LEO 阵列专属三状态路径参数，不复用 RFALT 的台架采样器。"""
    p, arr, lb = sim_cfg["physics"], sim_cfg["array"], sim_cfg["link_budget"]
    U = lambda a, b: float(rng.uniform(a, b))
    return {
        "Ea_eV": U(*p["Ea_eV_range"]),
        "Tj_base_C": U(*p["Tj_range_C"]),
        "Tj_amp_C": U(5.0, 20.0),
        "duty": U(*p["duty_cycle_range"]),
        "scan_az_deg": U(*arr["scan_az_deg_range"]),
        "scan_period_orbits": U(2.0, 8.0),
        "margin0_dB": U(*lb["initial_margin_dB_range"]),
        "dphi_max_deg": U(*p["phase_drift_deg_range"]),
        "seed_traj": int(rng.integers(0, 2**31)),
    }


def _grid_positions(grid: list[int], d_lambda: float) -> np.ndarray:
    nx, ny = grid
    ix = np.arange(nx) - (nx - 1) / 2.0
    iy = np.arange(ny) - (ny - 1) / 2.0
    gx, gy = np.meshgrid(ix, iy, indexing="xy")
    return np.stack([gx.ravel() * d_lambda, gy.ravel() * d_lambda], axis=1)


def _subarray_ids(grid: list[int], block: int = 4) -> np.ndarray:
    nx, ny = grid
    ix = np.arange(nx)
    iy = np.arange(ny)
    gx, gy = np.meshgrid(ix, iy, indexing="xy")
    sx = (gx // block).ravel()
    sy = (gy // block).ravel()
    return (sx + sy * (ny // block)).astype(int)


def _array_pattern(a, dphi_rad, pos_x, u_grid, u0):
    psi = a * np.exp(1j * dphi_rad)
    B = np.exp(1j * 2.0 * np.pi * np.outer(u_grid - u0, pos_x))
    return np.abs(psi @ B.T) ** 2


def _refine_peak(patt_row, theta_grid, i):
    if 0 < i < len(theta_grid) - 1:
        ym, y0, yp = patt_row[i - 1], patt_row[i], patt_row[i + 1]
        denom = (ym - 2 * y0 + yp)
        delta = 0.5 * (ym - yp) / denom if abs(denom) > 1e-30 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        return float(y0 - 0.25 * (ym - yp) * delta), float(theta_grid[i] + delta)
    return float(patt_row[i]), float(theta_grid[i])


def _subarray_reduce(values: np.ndarray, subarray_ids: np.ndarray, n_sa: int) -> np.ndarray:
    """将通道量聚合为子阵标签，形状为 (T, n_subarray)。"""
    return np.stack([values[:, subarray_ids == sid].mean(axis=1) for sid in range(n_sa)], axis=1)


def simulate_gan_state(params: dict, sim_cfg: dict, traj_rng: np.random.Generator):
    """LEO 任务耦合三状态 T/R 目标仿真。

    状态 ``d_perm/q_trap/r_th`` 在通道层积分，只在原始文件中作为 latent 保存；
    ``labels`` 是无测量噪声的子阵级 RF 性能标签。该积分器显式耦合 90 分钟
    轨道热循环、任务 duty/scan 变化和二维热点场，与 RFALT 集总台架模型独立。
    """
    p, arr, dist, lim = (sim_cfg["physics"], sim_cfg["array"], sim_cfg["disturbance"], sim_cfg["service_limits"])
    dt_out, dt_phys = float(sim_cfg["sample_period_s"]), float(sim_cfg.get("physics_dt_s", 300.0))
    t_end = float(sim_cfg["duration_years"]) * SEC_PER_YEAR
    n_ch, grid = int(arr["n_elements"]), arr["grid"]
    d_lambda = float(arr["element_spacing_lambda"])
    if dt_phys <= 0.0:
        raise ValueError("physics_dt_s 必须为正")
    n_substeps = max(1, int(np.ceil(dt_out / dt_phys)))
    sub_dt_s = dt_out / n_substeps
    sub_offsets_s = (np.arange(n_substeps) + 0.5) * sub_dt_s
    n_out = int(np.floor(t_end / dt_out))
    if n_out < 2:
        raise ValueError("仿真时长至少需要两个遥测窗口")

    pos2d = _grid_positions(grid, d_lambda)
    states = {name: np.empty((n_out, n_ch), dtype=np.float32) for name in ("d_perm", "q_trap", "r_th")}
    tj_out = np.empty(n_out, dtype=np.float64)
    duty_out = np.empty(n_out, dtype=np.float64)
    d_perm = np.zeros(n_ch, dtype=np.float64)
    q_trap = np.zeros(n_ch, dtype=np.float64)
    r_th = np.zeros(n_ch, dtype=np.float64)
    eta_phase = traj_rng.normal(0.0, 1.0, n_ch)
    grad_dir = traj_rng.normal(0.0, 1.0, 2)

    # 二维热点是目标阵列热耦合的一部分，不是 RFALT 单器件集总项。
    hotspot_c = np.zeros(n_ch, dtype=np.float64)
    hs = p.get("hotspot_field", {})
    if bool(hs.get("enabled", True)):
        for _ in range(int(traj_rng.integers(*hs["n_hotspots_range"], endpoint=True))):
            center = traj_rng.uniform(-1.0, 1.0, 2) * np.asarray(grid) * d_lambda * 0.5
            radius = float(traj_rng.uniform(*hs["radius_lambda_range"])) * d_lambda
            amplitude = float(traj_rng.uniform(*hs["Tj_boost_C_range"]))
            distance = np.linalg.norm(pos2d - center, axis=1)
            hotspot_c += amplitude * np.exp(-0.5 * (distance / max(radius, 1e-6)) ** 2)

    damage_cfg = p.get("damage_model", {})
    thermal_gain = float(damage_cfg.get("thermal_cycle_gain", 1.2))
    mission_gain = float(damage_cfg.get("duty_scan_gain", 0.8))
    rth_gain = float(damage_cfg.get("rth_feedback_gain", 4.0))
    # 每个遥测窗按 physics_dt_s 构造真实物理子步。永久/热阻状态使用全部
    # 子步增量之和，陷阱态逐子步解析推进；故改变 physics_dt_s 会改变积分求积精度，
    # 而输出时间轴始终由 sample_period_s 定义。
    q_scalar = 0.0
    for wi in range(n_out):
        sub_time_s = wi * dt_out + sub_offsets_s
        orbit = np.sin(2.0 * np.pi * sub_time_s / ORBIT_THERMAL_PERIOD_S)
        scan = np.sin(2.0 * np.pi * sub_time_s / (params["scan_period_orbits"] * ORBIT_THERMAL_PERIOD_S))
        duty_steps = np.clip(params["duty"] * (0.80 + 0.20 * (scan + 1.0) / 2.0), 0.05, 0.95)
        scan_load = 1.0 + mission_gain * np.abs(scan)
        capture_steps = 2.2e-4 * duty_steps * (1.0 + 0.30 * np.abs(scan)) * thermal_gain
        q_steps = np.empty(n_substeps, dtype=np.float64)
        for si, capture in enumerate(capture_steps):
            q_steady = capture / 0.075
            q_scalar = q_steady + (q_scalar - q_steady) * np.exp(-0.075 * sub_dt_s / 3600.0)
            q_steps[si] = q_scalar
        t_j = params["Tj_base_C"] + hotspot_c + rth_gain * r_th
        # 轨道温度相位、任务扫描与 q_trap 均在子步网格求积；热点/R_th 仍是通道态。
        thermal_base = np.exp((t_j - 105.0) / 45.0)
        thermal_orbit = np.exp(params["Tj_amp_C"] * orbit / 45.0)
        exposure_h = (sub_dt_s / 3600.0) * np.sum(
            thermal_orbit * duty_steps * scan_load * (1.0 + 0.35 * q_steps))
        increment = 4.0e-8 * np.clip(thermal_base * exposure_h, 0.0, 12.0)
        q_trap.fill(q_scalar)
        d_perm += np.maximum(increment, 0.0)
        r_th += np.maximum(increment, 0.0) * 12.0
        states["d_perm"][wi] = d_perm
        states["q_trap"][wi] = q_trap
        states["r_th"][wi] = r_th
        tj_out[wi], duty_out[wi] = np.mean(t_j), float(np.mean(duty_steps))

    d_perm, q_trap, r_th = states["d_perm"], states["q_trap"], states["r_th"]
    gain_ch = 12.0 - 480.0 * d_perm - 1.30 * q_trap - 4.0 * r_th
    gradient = (pos2d[:, 0] * grad_dir[0] + pos2d[:, 1] * grad_dir[1])
    phase_ch = params["dphi_max_deg"] * (60.0 * d_perm + 0.75 * q_trap) * eta_phase[None, :]
    phase_ch += 0.25 * params["dphi_max_deg"] * d_perm * gradient[None, :]
    pout_ch = 42.0 + gain_ch
    pae_ch = np.clip(0.58 - 35.0 * d_perm - 0.11 * q_trap - 0.25 * r_th, 0.01, 0.85)
    p_ratio = np.power(10.0, (gain_ch - gain_ch[0]) / 10.0)
    dropout = d_perm >= (0.0055 + 0.0015 * traj_rng.random(n_ch))[None, :]
    amplitude = np.sqrt(np.clip(np.where(dropout, 0.0, p_ratio), 0.0, None))

    # 第三级复用现有阵列方向图与链路预算计算。
    u0 = np.sin(np.deg2rad(params["scan_az_deg"]))
    pattern = _array_pattern(amplitude, np.deg2rad(phase_ch), pos2d[:, 0], np.sin(np.deg2rad(THETA_COARSE_DEG)), u0)
    peak_idx = np.argmax(pattern, axis=1)
    peak_pow, theta_peak = np.empty(n_out), np.empty(n_out)
    for ti in range(n_out):
        peak_pow[ti], theta_peak[ti] = _refine_peak(pattern[ti], THETA_COARSE_DEG, int(peak_idx[ti]))
    du = 1.0 / (grid[0] * d_lambda)
    left, right = np.degrees(np.arcsin(np.clip([u0 - du, u0 + du], -0.99, 0.99)))
    side = pattern.copy()
    side[:, (THETA_COARSE_DEG >= left) & (THETA_COARSE_DEG <= right)] = 0.0
    sll_true = np.clip(10.0 * np.log10(np.max(side, axis=1) / (peak_pow + 1e-30) + 1e-30), -50.0, 0.0)
    g_true = 10.0 * np.log10(peak_pow / (n_ch * n_ch) + 1e-30)
    deirp = np.clip(g_true[0] - g_true, 0.0, 40.0)
    m_true = params["margin0_dB"] - deirp
    theta_true = theta_peak - params["scan_az_deg"]
    violate = (m_true <= 0.0) | (sll_true > float(lim["SLL_max_dB"])) | (np.abs(theta_true) > float(lim["theta_err_max_deg"]))
    label_fail = np.zeros(n_out, dtype=np.int8)
    eol_idx, failed = n_out - 1, False
    consecutive = int(lim.get("consecutive_windows", 1))
    hits = np.convolve(violate.astype(int), np.ones(consecutive, dtype=int), mode="full")[:n_out]
    if (hits >= consecutive).any():
        eol_idx, failed = int(np.argmax(hits >= consecutive)), True
        label_fail[eol_idx:] = 1

    nr = float(dist["noise_ratio"])
    df = pd.DataFrame({
        "t": np.arange(n_out, dtype=float) * dt_out,
        "damage": d_perm.mean(axis=1), "EIRP_norm": peak_pow / peak_pow[0],
        "G_array_dB": g_true + traj_rng.normal(0.0, nr, n_out), "G_array_dB_true": g_true,
        "SLL_dB": sll_true + traj_rng.normal(0.0, nr, n_out), "SLL_dB_true": sll_true,
        "theta_err_deg": theta_true + traj_rng.normal(0.0, nr * 2.0, n_out), "theta_err_deg_true": theta_true,
        "M_link_dB": m_true + traj_rng.normal(0.0, nr, n_out), "M_link_dB_true": m_true,
        "k_failed": dropout.sum(axis=1).astype(float), "RDS_drift": 1.0 + 3.0 * d_perm.mean(axis=1),
        "IDSS_ratio": np.clip(1.0 - 65.0 * d_perm.mean(axis=1) - .2 * q_trap.mean(axis=1), .1, 1.0),
        "gm_ratio": np.clip(1.0 - 40.0 * d_perm.mean(axis=1), .1, 1.0), "Tj": tj_out,
        "Tj_max": tj_out, "Tj_min": tj_out, "duty": duty_out, "label_fail": label_fail,
    })
    sa_ids = _subarray_ids(grid, block=int(arr.get("subarray_block", 4)))
    n_sa = int(sa_ids.max()) + 1
    sa = np.empty((n_out, n_sa, 8), dtype=np.float32)
    for sid in range(n_sa):
        m = sa_ids == sid
        p_eff = np.where(dropout[:, m], 0.0, p_ratio[:, m])
        sa[:, sid, 0] = p_eff.mean(axis=1)
        sa[:, sid, 1] = np.quantile(p_eff, .10, axis=1)
        sa[:, sid, 2] = np.clip(1.0 - 65.0 * d_perm[:, m] - .2 * q_trap[:, m], 0.0, 1.0).mean(axis=1)
        sa[:, sid, 3] = tj_out
        sa[:, sid, 4] = np.std(amplitude[:, m], axis=1)
        sa[:, sid, 5] = np.std(np.deg2rad(phase_ch[:, m]), axis=1)
        sa[:, sid, 6] = (~dropout[:, m]).mean(axis=1)
        sa[:, sid, 7] = np.quantile(d_perm[:, m], .90, axis=1)
    labels = {"gain_dB": _subarray_reduce(gain_ch, sa_ids, n_sa).astype(np.float32),
              "phase_deg": _subarray_reduce(phase_ch, sa_ids, n_sa).astype(np.float32),
              "Pout_dBm": _subarray_reduce(pout_ch, sa_ids, n_sa).astype(np.float32),
              "PAE": _subarray_reduce(pae_ch, sa_ids, n_sa).astype(np.float32)}
    return df, sa, eol_idx, failed, labels, states


def _write_target_group(group, trajectory) -> None:
    df, sa, eol, failed, labels, states, attrs = trajectory
    for column in df.columns:
        group.create_dataset(column, data=df[column].values)
    large_kwargs = {"chunks": True, "compression": "gzip", "compression_opts": 4, "shuffle": True}
    group.create_dataset("subarray_features", data=sa.astype(np.float32), **large_kwargs)
    for name, values in states.items():
        group.create_dataset(f"latent_{name}", data=np.asarray(values, np.float32), **large_kwargs)
    for name, values in labels.items():
        group.create_dataset(f"label_channel_{name}", data=np.asarray(values, np.float32), **large_kwargs)
    group.attrs["eol_idx"] = int(eol)
    group.attrs["failed"] = int(failed)
    group.attrs["label_level"] = "subarray"
    for key, value in attrs.items():
        group.attrs[key] = value


def write_target_raw_h5(trajectories, output_path: str | Path) -> None:
    """写出 LEO 目标原始文件；状态和 RF 标签独立于遥测列保存。"""
    import h5py
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["dynamics_id"] = DYNAMICS_ID
        h5.attrs["target_condition_schema"] = TARGET_CONDITION_SCHEMA
        for index, trajectory in enumerate(trajectories):
            _write_target_group(h5.create_group(f"traj_{index:03d}"), trajectory)


def simulate(params: dict, sim_cfg: dict, traj_rng: np.random.Generator,
             fault_cfg: dict | None = None):
    """单条轨迹仿真 -> (df, subarray_block, eol_idx, failed)。

    fault_cfg: 故障注入配置 (§5.7)。None 或 type="none" 时行为完全不变。
    """
    if sim_cfg.get("physics", {}).get("damage_path", "legacy_scalar") == "gan_state":
        return simulate_gan_state(params, sim_cfg, traj_rng)[:4]
    if bool(sim_cfg.get("physics", {}).get("subarray_dose", {}).get("enabled", False)):
        return _simulate_subdose(params, sim_cfg, traj_rng, fault_cfg=fault_cfg)
    dt_out = float(sim_cfg["sample_period_s"])
    dt_phys = float(sim_cfg.get("physics_dt_s", 300.0))
    T_years = float(sim_cfg["duration_years"])
    arr = sim_cfg["array"]
    dist = sim_cfg["disturbance"]
    lim = sim_cfg["service_limits"]
    p = sim_cfg["physics"]
    n_ch = int(arr["n_elements"])
    grid = arr["grid"]
    d_lambda = float(arr["element_spacing_lambda"])
    consec = int(lim.get("consecutive_windows", 1))

    t_end = T_years * SEC_PER_YEAR
    t_phys_s = np.arange(0, t_end + dt_phys, dt_phys)

    # ---------- 第一级: Arrhenius + Coffin–Manson 累积损伤 (P1 #2) ----------
    # dose_rate = duty · [ΔTj/ΔT_ref]^n (Coffin-Manson 热循环疲劳) · Arrhenius(Tj)
    # R_th(s)=R_th0·(1+a5·s) 正反馈: damage↑→R_th↑→Tj↑→加速 (末期加速, 两步近似)
    Ea = params["Ea_eV"]
    Tj_base_phys = (params["Tj_base_C"] + 273.15
                    + params["Tj_amp_C"] * np.sin(2 * np.pi * t_phys_s / ORBIT_THERMAL_PERIOD_S)
                    + params["Tj_drift_C_per_year"] * (t_phys_s / SEC_PER_YEAR))
    deltaTj = 2.0 * params["Tj_amp_C"]                  # 轨道热循环峰峰 (90min 日照/阴影)
    cm = p.get("damage_model", {}).get("coffin_manson", {})
    rth = p.get("damage_model", {}).get("rth_feedback", {})
    cm_on = bool(cm.get("enabled", True))
    rth_on = bool(rth.get("enabled", True))
    dT_ref = float(cm.get("deltaT_ref_K", 20.0))
    n_cm = float(cm.get("exponent_n", 3.0))
    a5 = float(rth.get("a5", 0.3))
    Trise = float(rth.get("Trise_per_Rth_K", 15.0))
    # 第一遍 (无反馈): 算 s0 用于 R_th 反馈量
    accel0 = np.exp(-Ea / K_B * (1.0 / Tj_base_phys - 1.0 / TREF_K))
    cmf0 = (deltaTj / dT_ref) ** n_cm if cm_on else 1.0
    eff_age0 = np.cumsum(params["duty"] * cmf0 * accel0) * dt_phys / SEC_PER_YEAR
    s0 = np.clip(eff_age0 / params["life_scale_years"], 0.0, 1.5)   # clip 1.5 防 R_th 反馈在已失效后失控爆炸
    # 第二遍 (R_th 正反馈): Tj↑ + ΔTj↑ → 重算 dose
    if rth_on:
        fb = a5 * s0                                        # 反馈量随 damage 增
        Tj_phys = Tj_base_phys + fb * Trise                 # 末期额外温升
        deltaTj_fb = deltaTj * (1.0 + 0.5 * fb)             # 振幅随 R_th 放大
        accel = np.exp(-Ea / K_B * (1.0 / Tj_phys - 1.0 / TREF_K))
        cmf = (deltaTj_fb / dT_ref) ** n_cm if cm_on else 1.0
        dose = params["duty"] * cmf * accel
    else:
        Tj_phys = Tj_base_phys
        dose = params["duty"] * cmf0 * accel0
    eff_age = np.cumsum(dose) * dt_phys / SEC_PER_YEAR
    # 6h 窗口聚合 (修复 GPT 十一: 不取起点瞬时同相位)
    steps_per_win = max(1, int(round(dt_out / dt_phys)))
    n_win = len(t_phys_s) // steps_per_win
    Tj_win = Tj_phys[:n_win * steps_per_win].reshape(n_win, steps_per_win)
    Tj_out = Tj_win.mean(axis=1)
    Tj_max_out = Tj_win.max(axis=1)
    Tj_min_out = Tj_win.min(axis=1)
    eff_age_out = eff_age[steps_per_win - 1::steps_per_win][:n_win]
    t_out_s = (np.arange(1, n_win + 1) * steps_per_win - 1) * dt_phys
    n_out = n_win
    # P1 #2: s=D/D_EOL 线性 (用户公式, 去掉旧 time_exp_m 幂律); R_th 反馈末期超线性直接体现在 damage
    damage = np.clip(eff_age_out / params["life_scale_years"], 0.0, None)
    f = damage

    # ---------- 第二级: GaN 参数漂移 + 空间相位梯度 ----------
    Delta_R = params["Delta_R"]
    pos2d = _grid_positions(grid, d_lambda)
    eta_phi = traj_rng.normal(0.0, 1.0, n_ch)
    eta_R = traj_rng.normal(1.0, 0.05, n_ch)
    # 二维相关热点场 (P1 #3): 热点区 Tj 高 → Arrhenius 加速 → damage 聚簇失效
    hs = p.get("hotspot_field", {})
    Tj_ref_K = params["Tj_base_C"] + 273.15
    if bool(hs.get("enabled", True)):
        n_hs = int(traj_rng.integers(*hs["n_hotspots_range"], endpoint=True))
        damage_boost = np.ones(n_ch)
        for _ in range(n_hs):
            cx = float(traj_rng.uniform(-1, 1)) * (grid[0] * d_lambda) * 0.5
            cy = float(traj_rng.uniform(-1, 1)) * (grid[1] * d_lambda) * 0.5
            rad = float(traj_rng.uniform(*hs["radius_lambda_range"])) * d_lambda
            boost_K = float(traj_rng.uniform(*hs["Tj_boost_C_range"]))
            d2 = np.sqrt((pos2d[:, 0] - cx) ** 2 + (pos2d[:, 1] - cy) ** 2)
            Tj_offset = boost_K * np.exp(-0.5 * (d2 / rad) ** 2)   # 高斯衰减
            # Arrhenius 加速比: exp(Ea/kB·(1/T_ref − 1/(T_ref+ΔT)))
            damage_boost *= np.exp(Ea / K_B * (1.0 / Tj_ref_K - 1.0 / (Tj_ref_K + Tj_offset)))
        damage_boost = np.clip(damage_boost, 1.0, 5.0)   # 限单通道加速 ≤5x, 防极端聚簇致方向图崩溃
        f_ch = np.clip(f[:, None] * damage_boost[None, :] *
                       (1.0 + 0.10 * traj_rng.normal(0, 1, n_ch)[None, :]), 0, None)
    else:
        f_ch = np.clip(f[:, None] * (1.0 + 0.15 * traj_rng.normal(0, 1, n_ch)[None, :]), 0, None)
    # 终末通道退出 + 2D 聚簇
    dropout_mask = np.zeros_like(f_ch, dtype=bool)
    if p["channel_dropout_weibull"]["enabled"]:
        thr = np.clip(traj_rng.weibull(params["weibull_beta"], n_ch) * 0.3 + 0.8, 0.6, 1.5)
        if traj_rng.random() < float(dist["spatial_cluster_prob"]):
            seed_ch = int(traj_rng.integers(0, n_ch))
            d2 = np.sqrt((pos2d[:, 0] - pos2d[seed_ch, 0]) ** 2 + (pos2d[:, 1] - pos2d[seed_ch, 1]) ** 2)
            thr[d2 < 1.0] *= 0.85
        dropout_mask = f_ch > thr[None, :]
    RDS_ratio = 1.0 + (Delta_R - 1.0) * f_ch
    IDSS_ratio = np.clip(1.0 - params["decay_I"] * f_ch, 0.1, 1.0)
    gm_ratio = np.clip(1.0 - params["decay_g"] * f_ch, 0.1, 1.0)
    P_ratio = IDSS_ratio * gm_ratio / RDS_ratio
    a = np.sqrt(np.clip(P_ratio, 0.0, None)) * eta_R[None, :]
    # 相位: 独立零均值 + 空间相关梯度 (修复 GPT 十三: 形成持续指向偏移)
    grad_dir = traj_rng.normal(0.0, 1.0, 2)
    grad_amp = np.deg2rad(params["dphi_max_deg"]) * 0.3
    dphi_grad = grad_amp * np.outer(f, grad_dir[0] * pos2d[:, 0] + grad_dir[1] * pos2d[:, 1])
    dphi_rad = np.deg2rad(params["dphi_max_deg"]) * f_ch * eta_phi[None, :] + dphi_grad
    a = np.where(dropout_mask, 0.0, a)

    # ---------- 第三级: 阵列方向图 ----------
    pos_x = pos2d[:, 0]
    u0 = np.sin(np.deg2rad(params["scan_az_deg"]))
    patt = _array_pattern(a, dphi_rad, pos_x, np.sin(np.deg2rad(THETA_COARSE_DEG)), u0)
    peak_idx = np.argmax(patt, axis=1)
    peak_pow = np.empty(n_out)
    theta_peak = np.empty(n_out)
    for ti in range(n_out):
        pw, th = _refine_peak(patt[ti], THETA_COARSE_DEG, int(peak_idx[ti]))
        peak_pow[ti] = pw
        theta_peak[ti] = th
    # SLL 主瓣窗 = 左右首零点之间 (修复 GPT 十二)
    du = 1.0 / (grid[0] * d_lambda)
    th_left = np.degrees(np.arcsin(np.clip(u0 - du, -0.99, 0.99)))
    th_right = np.degrees(np.arcsin(np.clip(u0 + du, -0.99, 0.99)))
    main_mask = (THETA_COARSE_DEG >= th_left) & (THETA_COARSE_DEG <= th_right)
    patt_sl = patt.copy()
    patt_sl[:, main_mask] = 0.0
    sll_lin = np.max(patt_sl, axis=1) / (peak_pow + 1e-30)
    SLL_true = np.clip(10.0 * np.log10(sll_lin + 1e-30), -50.0, 0.0)   # clip -50 防方向图崩溃致数值失真

    G_array_true = 10.0 * np.log10(peak_pow / (n_ch * n_ch) + 1e-30)
    dEIRP_true = np.clip(G_array_true[0] - G_array_true, 0.0, 40.0)
    M_link_true = params["margin0_dB"] - dEIRP_true
    EIRP_norm = peak_pow / peak_pow[0]
    theta_err_true = theta_peak - params["scan_az_deg"]

    violate = (M_link_true <= 0.0) | (SLL_true > lim["SLL_max_dB"]) | \
              (np.abs(theta_err_true) > lim["theta_err_max_deg"])
    label_fail = np.zeros(n_out, dtype=np.int8)
    eol_idx = n_out - 1
    failed = False
    if consec <= 1:
        if violate.any():
            eol_idx = int(np.argmax(violate)); label_fail[eol_idx:] = 1; failed = True
    else:
        cs = np.convolve(violate.astype(int), np.ones(consec, dtype=int), mode="full")[:n_out]
        if (cs >= consec).any():
            eol_idx = int(np.argmax(cs >= consec)); label_fail[eol_idx:] = 1; failed = True

    # 噪声 (obs)
    nr = float(dist["noise_ratio"])
    rn = traj_rng.normal
    G_array_obs = G_array_true + rn(0, nr, n_out)
    SLL_obs = SLL_true + rn(0, nr, n_out)
    theta_err_obs = theta_err_true + rn(0, nr * 2.0, n_out)
    M_link_obs = M_link_true + rn(0, nr, n_out)
    k_failed = np.sum(a <= 1e-9, axis=1).astype(float)

    df = pd.DataFrame({
        "t": t_out_s, "damage": f, "EIRP_norm": EIRP_norm,
        "G_array_dB": G_array_obs, "G_array_dB_true": G_array_true,
        "SLL_dB": SLL_obs, "SLL_dB_true": SLL_true,
        "theta_err_deg": theta_err_obs, "theta_err_deg_true": theta_err_true,
        "M_link_dB": M_link_obs, "M_link_dB_true": M_link_true,
        "k_failed": k_failed, "RDS_drift": RDS_ratio.mean(axis=1),
        "IDSS_ratio": IDSS_ratio.mean(axis=1), "gm_ratio": gm_ratio.mean(axis=1),
        "Tj": Tj_out, "Tj_max": Tj_max_out, "Tj_min": Tj_min_out,
        "duty": np.full(n_out, params["duty"]), "label_fail": label_fail,
    })

    # 子阵级 (dropout 入功率, 修复 GPT 十四)
    sa_ids = _subarray_ids(grid, block=int(arr.get("subarray_block", 4)))
    n_sa = int(sa_ids.max()) + 1
    P_effective = np.where(dropout_mask, 0.0, P_ratio)
    IDSS_eff = np.where(dropout_mask, 0.0, IDSS_ratio)
    sa_feat = np.zeros((n_out, n_sa, 8), dtype=np.float32)
    for s in range(n_sa):
        msk = sa_ids == s
        P_s = P_effective[:, msk]
        sa_feat[:, s, 0] = P_s.mean(axis=1)
        sa_feat[:, s, 1] = np.quantile(P_s, 0.10, axis=1)
        sa_feat[:, s, 2] = IDSS_eff[:, msk].mean(axis=1)
        sa_feat[:, s, 3] = Tj_out
        sa_feat[:, s, 4] = np.std(np.where(dropout_mask[:, msk], 0.0, a[:, msk]), axis=1)
        sa_feat[:, s, 5] = np.std(dphi_rad[:, msk], axis=1)
        sa_feat[:, s, 6] = (~dropout_mask[:, msk]).mean(axis=1).astype(float)
        sa_feat[:, s, 7] = np.quantile(f_ch[:, msk], 0.90, axis=1)
    return df, sa_feat, eol_idx, failed, None


def _simulate_subdose(params: dict, sim_cfg: dict, traj_rng: np.random.Generator,
                      fault_cfg: dict | None = None):
    """子阵级独立损伤积分路径 (T1/M1, subarray_dose.enabled=true).

    与 simulate() 标量路径的区别: 每子阵独立 Tj_offset_s + life_scale_s, 独立两遍
    Arrhenius+Coffin-Manson 积分 → f_sub(T,16); 元件级 f_ch=f_sub[:,s(i)]*c_elem (静态缩放)。
    第三级 (阵列方向图/服务寿命/df/sa_feat) 与 simulate() 逐字一致, 保证 M4 物理孪生可复现。
    返回第 5 项 twin dict (twin_only, 不进模型输入), 供 M4 服务层前推。

    fault_cfg: 故障注入配置 (§5.7)。None 或 type="none" 时函数行为完全不变 (逐位复现)。
    """
    dt_out = float(sim_cfg["sample_period_s"])
    dt_phys = float(sim_cfg.get("physics_dt_s", 300.0))
    T_years = float(sim_cfg["duration_years"])
    arr = sim_cfg["array"]
    dist = sim_cfg["disturbance"]
    lim = sim_cfg["service_limits"]
    p = sim_cfg["physics"]
    n_ch = int(arr["n_elements"])
    grid = arr["grid"]
    d_lambda = float(arr["element_spacing_lambda"])
    consec = int(lim.get("consecutive_windows", 1))

    t_end = T_years * SEC_PER_YEAR
    t_phys_s = np.arange(0, t_end + dt_phys, dt_phys)

    # ---------- 公共物理量 (无 RNG) ----------
    Ea = params["Ea_eV"]
    Tj_base_phys = (params["Tj_base_C"] + 273.15
                    + params["Tj_amp_C"] * np.sin(2 * np.pi * t_phys_s / ORBIT_THERMAL_PERIOD_S)
                    + params["Tj_drift_C_per_year"] * (t_phys_s / SEC_PER_YEAR))
    deltaTj = 2.0 * params["Tj_amp_C"]
    cm = p.get("damage_model", {}).get("coffin_manson", {})
    rth = p.get("damage_model", {}).get("rth_feedback", {})
    cm_on = bool(cm.get("enabled", True))
    rth_on = bool(rth.get("enabled", True))
    dT_ref = float(cm.get("deltaT_ref_K", 20.0))
    n_cm = float(cm.get("exponent_n", 3.0))
    a5 = float(rth.get("a5", 0.3))
    Trise = float(rth.get("Trise_per_Rth_K", 15.0))
    cmf0 = (deltaTj / dT_ref) ** n_cm if cm_on else 1.0
    steps_per_win = max(1, int(round(dt_out / dt_phys)))
    n_win = len(t_phys_s) // steps_per_win
    n_out = n_win
    t_out_s = (np.arange(1, n_win + 1) * steps_per_win - 1) * dt_phys
    pos2d = _grid_positions(grid, d_lambda)
    sa_ids = _subarray_ids(grid, block=int(arr.get("subarray_block", 4)))
    n_sa = int(sa_ids.max()) + 1
    Tj_ref_K = params["Tj_base_C"] + 273.15
    hs = p.get("hotspot_field", {})
    sigma_sub = float(p.get("subarray_dose", {}).get("sigma_sub", 0.20))

    # ---------- 静态散布采样 (落盘 twin_*) ----------
    eta_phi = traj_rng.normal(0.0, 1.0, n_ch)            # 相位随机系数 (twin_eta_phi)
    eta_R = traj_rng.normal(1.0, 0.05, n_ch)             # 幅度制造散布 (twin_eta_R)
    grad_dir = traj_rng.normal(0.0, 1.0, 2)              # 相位空间梯度方向 (twin_grad_dir)
    xi_elem = traj_rng.normal(0.0, 1.0, n_ch)            # 元件级静态散布 (与 t 无关)
    # 热点场: 每元件温升偏移 → Arrhenius 加速比
    Tj_offset_elem = np.zeros(n_ch)
    if bool(hs.get("enabled", True)):
        n_hs = int(traj_rng.integers(*hs["n_hotspots_range"], endpoint=True))
        for _ in range(n_hs):
            cx = float(traj_rng.uniform(-1, 1)) * (grid[0] * d_lambda) * 0.5
            cy = float(traj_rng.uniform(-1, 1)) * (grid[1] * d_lambda) * 0.5
            rad = float(traj_rng.uniform(*hs["radius_lambda_range"])) * d_lambda
            boost_K = float(traj_rng.uniform(*hs["Tj_boost_C_range"]))
            d2 = np.sqrt((pos2d[:, 0] - cx) ** 2 + (pos2d[:, 1] - cy) ** 2)
            Tj_offset_elem += boost_K * np.exp(-0.5 * (d2 / rad) ** 2)
    damage_boost = np.clip(
        np.exp(Ea / K_B * (1.0 / Tj_ref_K - 1.0 / (Tj_ref_K + Tj_offset_elem))), 1.0, 5.0)

    # ---------- 第一级: 子阵级独立 dose 积分 (每子阵独立 Tj_offset + life_scale) ----------
    Tj_offset_s = np.array([Tj_offset_elem[sa_ids == s].mean() for s in range(n_sa)])
    life_scale_s = params["life_scale_years"] * np.exp(
        sigma_sub * traj_rng.normal(0.0, 1.0, n_sa))                                   # (16,)
    Tj_base_s = Tj_base_phys[:, None] + Tj_offset_s[None, :]                           # (n_phys, 16)
    accel0_s = np.exp(-Ea / K_B * (1.0 / Tj_base_s - 1.0 / TREF_K))
    eff_age0_s = np.cumsum(params["duty"] * cmf0 * accel0_s, axis=0) * dt_phys / SEC_PER_YEAR
    s0_s = np.clip(eff_age0_s / life_scale_s[None, :], 0.0, 1.5)
    if rth_on:
        # F1 (rth_step) / F2 (thermal_bias): 仅在注入时点之后偏离标称
        # ft="none" 时走原始标量路径, 保证逐位复现
        ft = fault_cfg.get("type", "none") if fault_cfg else "none"
        fb_s = a5 * s0_s                            # 标量路径 (注入前逐位不变)

        if ft == "rth_step":
            # 时变注入: 注入时点之后目标子阵的反馈系数增大 (注入前完全不变)
            si = fault_cfg["start_idx_phys"]
            fb_s[si:, fault_cfg["sub_id"]] *= fault_cfg["r_th_mult"]

        if ft == "thermal_bias":
            si = fault_cfg["start_idx_phys"]
            rw = max(1, fault_cfg.get("ramp_windows", 10)) * steps_per_win
            rl = min(len(t_phys_s) - si, rw)
            tj_bias = np.zeros(len(t_phys_s))
            if rl > 0:
                tj_bias[si:si+rl] = np.linspace(0, fault_cfg["tj_offset_K"], rl)
                tj_bias[si+rl:] = fault_cfg["tj_offset_K"]
            Tj_phys_s = Tj_base_s + fb_s * Trise + tj_bias[:, None]
        else:
            Tj_phys_s = Tj_base_s + fb_s * Trise    # 原始路径 (逐位不变)

        deltaTj_fb_s = deltaTj * (1.0 + 0.5 * fb_s)
        accel_s = np.exp(-Ea / K_B * (1.0 / Tj_phys_s - 1.0 / TREF_K))
        cmf_s = (deltaTj_fb_s / dT_ref) ** n_cm if cm_on else 1.0
        dose_s = params["duty"] * cmf_s * accel_s
        Tj_for_out = Tj_phys_s
    else:
        dose_s = params["duty"] * cmf0 * accel0_s
        Tj_for_out = Tj_base_s
    eff_age_s = np.cumsum(dose_s, axis=0) * dt_phys / SEC_PER_YEAR                       # (n_phys, 16)
    eff_age_out_s = eff_age_s[steps_per_win - 1::steps_per_win][:n_win]                 # (n_win, 16)
    f_sub = np.clip(eff_age_out_s / life_scale_s[None, :], 0.0, None)                   # (n_out, 16) 子阵损伤真值
    Tj_win_s = Tj_for_out[:n_win * steps_per_win].reshape(n_win, steps_per_win, n_sa)
    Tj_out = Tj_win_s.mean(axis=(1, 2))
    Tj_max_out = Tj_win_s.max(axis=(1, 2))
    Tj_min_out = Tj_win_s.min(axis=(1, 2))
    damage = f_sub.mean(axis=1)                         # 轨迹级标量 (df 输出兼容)
    f = damage                                           # dphi_grad 用轨迹级标量 (同 simulate)
    # 元件级静态缩放 c_elem (与 t 无关): boost 归一到子阵均值 × 元件散布
    boost_mean_per_sub = np.array([damage_boost[sa_ids == s].mean() for s in range(n_sa)])
    c_elem = damage_boost / boost_mean_per_sub[sa_ids] * (1.0 + 0.10 * xi_elem)          # (256,)
    f_ch = np.clip(f_sub[:, sa_ids] * c_elem[None, :], 0, None)                         # (n_out, 256)

    # ---------- 第二级: GaN 参数漂移 + 空间相位梯度 ----------
    Delta_R = params["Delta_R"]
    RDS_ratio = 1.0 + (Delta_R - 1.0) * f_ch
    IDSS_ratio = np.clip(1.0 - params["decay_I"] * f_ch, 0.1, 1.0)
    gm_ratio = np.clip(1.0 - params["decay_g"] * f_ch, 0.1, 1.0)
    P_ratio = IDSS_ratio * gm_ratio / RDS_ratio
    a = np.sqrt(np.clip(P_ratio, 0.0, None)) * eta_R[None, :]
    grad_amp = np.deg2rad(params["dphi_max_deg"]) * 0.3
    dphi_grad = grad_amp * np.outer(f, grad_dir[0] * pos2d[:, 0] + grad_dir[1] * pos2d[:, 1])
    dphi_rad = np.deg2rad(params["dphi_max_deg"]) * f_ch * eta_phi[None, :] + dphi_grad

    # F4 (cal_freeze): 注入时点后相位残差放大 (校准环路失效, 补偿不再压制残差)
    ft = fault_cfg.get("type", "none") if fault_cfg else "none"
    if ft == "cal_freeze":
        dphi_rad[fault_cfg["start_win"]:] *= fault_cfg["phase_residual_mult"]

    dropout_mask = np.zeros_like(f_ch, dtype=bool)
    dropout_thr = np.zeros(n_ch)
    if p["channel_dropout_weibull"]["enabled"]:
        thr = np.clip(traj_rng.weibull(params["weibull_beta"], n_ch) * 0.3 + 0.8, 0.6, 1.5)
        if traj_rng.random() < float(dist["spatial_cluster_prob"]):
            seed_ch = int(traj_rng.integers(0, n_ch))
            d2 = np.sqrt((pos2d[:, 0] - pos2d[seed_ch, 0]) ** 2 + (pos2d[:, 1] - pos2d[seed_ch, 1]) ** 2)
            thr[d2 < 1.0] *= 0.85
        dropout_thr = thr
        dropout_mask = f_ch > thr[None, :]

    # F3 (channel_open): 强制子阵内 n_channels 通道提前 dropout (与累积损伤解耦的灾难失效)
    if ft == "channel_open":
        target_ch = np.where(sa_ids == fault_cfg["sub_id"])[0][:fault_cfg["n_channels"]]
        dropout_mask[fault_cfg["start_win"]:, target_ch] = True

    a = np.where(dropout_mask, 0.0, a)

    # ---------- 第三级: 阵列方向图 (与 simulate() 严格一致, M4 一致性前提) ----------
    pos_x = pos2d[:, 0]
    u0 = np.sin(np.deg2rad(params["scan_az_deg"]))
    patt = _array_pattern(a, dphi_rad, pos_x, np.sin(np.deg2rad(THETA_COARSE_DEG)), u0)
    peak_idx = np.argmax(patt, axis=1)
    peak_pow = np.empty(n_out)
    theta_peak = np.empty(n_out)
    for ti in range(n_out):
        pw, th = _refine_peak(patt[ti], THETA_COARSE_DEG, int(peak_idx[ti]))
        peak_pow[ti] = pw
        theta_peak[ti] = th
    du = 1.0 / (grid[0] * d_lambda)
    th_left = np.degrees(np.arcsin(np.clip(u0 - du, -0.99, 0.99)))
    th_right = np.degrees(np.arcsin(np.clip(u0 + du, -0.99, 0.99)))
    main_mask = (THETA_COARSE_DEG >= th_left) & (THETA_COARSE_DEG <= th_right)
    patt_sl = patt.copy()
    patt_sl[:, main_mask] = 0.0
    sll_lin = np.max(patt_sl, axis=1) / (peak_pow + 1e-30)
    SLL_true = np.clip(10.0 * np.log10(sll_lin + 1e-30), -50.0, 0.0)
    G_array_true = 10.0 * np.log10(peak_pow / (n_ch * n_ch) + 1e-30)
    dEIRP_true = np.clip(G_array_true[0] - G_array_true, 0.0, 40.0)
    M_link_true = params["margin0_dB"] - dEIRP_true
    EIRP_norm = peak_pow / peak_pow[0]
    theta_err_true = theta_peak - params["scan_az_deg"]
    violate = (M_link_true <= 0.0) | (SLL_true > lim["SLL_max_dB"]) | \
              (np.abs(theta_err_true) > lim["theta_err_max_deg"])
    label_fail = np.zeros(n_out, dtype=np.int8)
    eol_idx = n_out - 1
    failed = False
    if consec <= 1:
        if violate.any():
            eol_idx = int(np.argmax(violate)); label_fail[eol_idx:] = 1; failed = True
    else:
        cs = np.convolve(violate.astype(int), np.ones(consec, dtype=int), mode="full")[:n_out]
        if (cs >= consec).any():
            eol_idx = int(np.argmax(cs >= consec)); label_fail[eol_idx:] = 1; failed = True

    # ---------- 噪声 + df/sa_feat 组装 (与 simulate() 一致) ----------
    nr = float(dist["noise_ratio"])
    rn = traj_rng.normal
    G_array_obs = G_array_true + rn(0, nr, n_out)
    SLL_obs = SLL_true + rn(0, nr, n_out)
    theta_err_obs = theta_err_true + rn(0, nr * 2.0, n_out)
    M_link_obs = M_link_true + rn(0, nr, n_out)
    k_failed = np.sum(a <= 1e-9, axis=1).astype(float)
    df = pd.DataFrame({
        "t": t_out_s, "damage": f, "EIRP_norm": EIRP_norm,
        "G_array_dB": G_array_obs + 0.0, "G_array_dB_true": G_array_true,
        "SLL_dB": SLL_obs + 0.0, "SLL_dB_true": SLL_true,
        "theta_err_deg": theta_err_obs + 0.0, "theta_err_deg_true": theta_err_true,
        "M_link_dB": M_link_obs + 0.0, "M_link_dB_true": M_link_true,
        "k_failed": k_failed, "RDS_drift": RDS_ratio.mean(axis=1),
        "IDSS_ratio": IDSS_ratio.mean(axis=1), "gm_ratio": gm_ratio.mean(axis=1),
        "Tj": Tj_out, "Tj_max": Tj_max_out, "Tj_min": Tj_min_out,
        "duty": np.full(n_out, params["duty"]), "label_fail": label_fail,
    })
    sa_feat = np.zeros((n_out, n_sa, 8), dtype=np.float32)
    for s in range(n_sa):
        msk = sa_ids == s
        P_s = np.where(dropout_mask[:, msk], 0.0, P_ratio[:, msk])
        sa_feat[:, s, 0] = P_s.mean(axis=1)
        sa_feat[:, s, 1] = np.quantile(P_s, 0.10, axis=1)
        sa_feat[:, s, 2] = np.where(dropout_mask[:, msk], 0.0, IDSS_ratio[:, msk]).mean(axis=1)
        sa_feat[:, s, 3] = Tj_out
        sa_feat[:, s, 4] = np.std(np.where(dropout_mask[:, msk], 0.0, a[:, msk]), axis=1)
        sa_feat[:, s, 5] = np.std(dphi_rad[:, msk], axis=1)
        sa_feat[:, s, 6] = (~dropout_mask[:, msk]).mean(axis=1).astype(float)
        sa_feat[:, s, 7] = np.quantile(f_ch[:, msk], 0.90, axis=1)

    # ---------- 孪生静态量 (twin_only, 不进模型输入 x; 供 M4 物理孪生前推) ----------
    twin = {
        "latent_sub_damage": f_sub.astype(np.float64),       # (n_out, 16) 子阵损伤真值 (float64 内部精度; 落盘 main 转 float32)
        "twin_c_elem": c_elem.astype(np.float64),            # (256,) 元件相对子阵静态缩放
        "twin_eta_R": eta_R.astype(np.float64),              # (256,) 幅度散布
        "twin_eta_phi": eta_phi.astype(np.float64),          # (256,) 相位随机系数
        "twin_dropout_thr": dropout_thr.astype(np.float64),  # (256,) Weibull 退出阈值 (含聚簇修正)
        "twin_grad_dir": grad_dir.astype(np.float64),        # (2,) 相位空间梯度方向
        "twin_subarray_ids": sa_ids.astype(np.int32),        # (256,) 元件→子阵映射
        "__verify_f_ch": f_ch.astype(np.float64),            # 内部验证用 (重建一致性), 下划线前缀不落盘
    }
    return df, sa_feat, eol_idx, failed, twin


def _build_fault_cfg(cfg: dict, fault_type: str, traj_id: int,
                     eol_nom: int | None, n_win: int, steps_per_win: int,
                     n_sa: int = 16) -> dict | None:
    """从 config 构建故障注入参数 (§5.7)。注入时点锚在标称 EOL 上, 删失回退到 horizon。

    幅值完全由 config 决定, 不含 RNG。给定 (traj_id, fault_type) 完全确定。
    """
    if fault_type == "none":
        return None
    fi_cfg = cfg.get("fault_injection", {})
    type_cfg = fi_cfg.get("types", {}).get(fault_type, {})
    start_frac = float(type_cfg.get("start_frac", 0.45))

    # 注入时点锚在标称 EOL 上; 删失 (eol_nom is None/0) 回退到 horizon 同比例
    anchor_eol = eol_nom if eol_nom and eol_nom > 0 else n_win
    start_win = max(1, min(int(start_frac * anchor_eol), n_win - 1))

    fc = dict(type=fault_type, start_win=start_win,
              start_idx_phys=start_win * steps_per_win)

    if fault_type == "rth_step":
        fc["sub_id"] = (traj_id + int(type_cfg.get("sub_offset", 0))) % n_sa
        fc["r_th_mult"] = float(type_cfg.get("r_th_mult", 1.6))
    elif fault_type == "thermal_bias":
        fc["tj_offset_K"] = float(type_cfg.get("tj_offset_K", 8.0))
        fc["ramp_windows"] = int(type_cfg.get("ramp_windows", 10))
    elif fault_type == "channel_open":
        fc["sub_id"] = (traj_id + int(type_cfg.get("sub_offset", 7))) % n_sa
        fc["n_channels"] = int(type_cfg.get("n_channels", 8))
    elif fault_type == "cal_freeze":
        fc["phase_residual_mult"] = float(type_cfg.get("phase_residual_mult", 4.0))
    return fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--n_traj", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--subdose", default="auto", choices=["on", "off", "auto"],
                    help="子阵级独立损伤开关: on=sim_v2 / off=sim_v1(旧标量) / auto=跟config; "
                         "用于不改 config 直接生成 sim_v1 (cross_level_transfer 复跑)")
    ap.add_argument("--inject", default="none",
                    choices=["none", "rth_step", "thermal_bias", "channel_open", "cal_freeze"],
                    help="故障注入类型 (§5.7): none=标称 / rth_step=F1 / thermal_bias=F2 / "
                         "channel_open=F3 / cal_freeze=F4")
    ap.add_argument("--fault-ids", default=None,
                    help="注入故障的轨迹 ID 列表 (逗号分隔, 如 '0,1,...,14'); 省略=全部注入")
    ap.add_argument("--fault-eol-file", default=None,
                    help="JSON 文件: {traj_id_str: eol_idx} 标称轨迹 EOL, 用于锚定注入时点")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])
    sim_cfg = dict(cfg["sim"])
    if args.fast:
        sim_cfg.update(n_traj=4, duration_years=1.0, sample_period_s=43200.0, physics_dt_s=900.0)
    n_traj = args.n_traj or int(sim_cfg["n_traj"])
    rng = np.random.default_rng(seed)
    damage_path = sim_cfg.get("physics", {}).get("damage_path", "legacy_scalar")
    if damage_path == "gan_state":
        target_cfg = cfg.get("target", {})
        raw_path = ROOT / (args.out or target_cfg.get("raw_path", "data/simulated/phased_array_gan/target_raw.h5"))
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        summaries, params_list = [], []
        h = hashlib.sha256()
        with h5py.File(raw_path, "w") as f:
            f.attrs["dynamics_id"] = DYNAMICS_ID
            f.attrs["target_condition_schema"] = TARGET_CONDITION_SCHEMA
            for i in range(n_traj):
                params = sample_gan_target_params(rng, sim_cfg)
                result = simulate_gan_state(params, sim_cfg, np.random.default_rng(params["seed_traj"]))
                df, _, eol, failed, _, _ = result
                _write_target_group(f.create_group(f"traj_{i:03d}"), (*result, params))
                params_list.append(params)
                h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
                summaries.append((failed, eol, len(df)))
        raw_path.with_name("params.json").write_text(json.dumps(params_list, indent=2), encoding="utf-8")
        n_fail = sum(failed for failed, _, _ in summaries)
        print(f">> gan_state 生成 {n_traj} 条 LEO 目标轨迹 -> {raw_path} (dynamics_id={DYNAMICS_ID})")
        print(f">> 失效轨迹 {n_fail}/{n_traj} ({100*n_fail/n_traj:.0f}%)")
        if args.hash:
            print(f">> repro hash (seed={seed}, n_traj={n_traj}): {h.hexdigest()[:16]}")
        return

    _subdose_cfg = bool(sim_cfg.get("physics", {}).get("subarray_dose", {}).get("enabled", False))
    subdose_on = {"auto": _subdose_cfg, "on": True, "off": False}[args.subdose]
    # --subdose 覆盖: 同步写入 sim_cfg, 让 simulate() 内部也走对应路径
    # (否则只改输出路径 sim_v1 但数据仍是 sim_v2 子阵损伤 → hash 不变, cross_level 复跑失效)
    sim_cfg.setdefault("physics", {}).setdefault("subarray_dose", {})["enabled"] = subdose_on
    if args.subdose != "auto":
        print(f">> --subdose {args.subdose}: subarray_dose {'on' if subdose_on else 'off'} "
              f"(config={_subdose_cfg})")
    sim_tag = "sim_v2" if subdose_on else "sim_v1"       # v2=子阵级损伤 (DYNAMICS_ID 变, 旧下游 schema 拒绝)
    dyn_id = "phased_array_subdose_v2" if subdose_on else DYNAMICS_ID

    # 故障注入时点参数 (§5.7)
    _dt_out = float(sim_cfg["sample_period_s"])
    _dt_phys = float(sim_cfg.get("physics_dt_s", 300.0))
    _spw = max(1, int(round(_dt_out / _dt_phys)))
    _n_win = int(float(sim_cfg["duration_years"]) * SEC_PER_YEAR / _dt_phys) // _spw
    fault_ids_set: set[int] = set()
    eol_map: dict[int, int] = {}
    if args.inject != "none":
        if args.fault_ids:
            fault_ids_set = set(int(x) for x in args.fault_ids.split(",") if x.strip())
        else:
            _n_per = int(cfg.get("fault_injection", {}).get("n_per_type", 15))
            fault_ids_set = set(range(min(_n_per, n_traj)))
        if args.fault_eol_file:
            eol_map = {int(k): (int(v) if v is not None else None) for k, v in
                       json.loads(Path(args.fault_eol_file).read_text(encoding="utf-8")).items()}
        print(f">> --inject {args.inject}: {len(fault_ids_set)} 条故障轨迹")

    _fault_tag = f"fault_{args.inject}" if args.inject != "none" else sim_tag
    out_dir = ROOT / (args.out or f"data/simulated/phased_array/{_fault_tag}/seed_{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, params_list = [], []
    h = hashlib.sha256()
    with h5py.File(out_dir / "phased_array_all.h5", "w") as f:    # 流式写盘 (GPT 十六)
        f.attrs["dynamics_id"] = dyn_id
        f.attrs["target_condition_schema"] = TARGET_CONDITION_SCHEMA
        for i in range(n_traj):
            params = sample_params(rng, sim_cfg)
            traj_rng = np.random.default_rng(params["seed_traj"])

            fault_cfg_i = None
            if args.inject != "none" and i in fault_ids_set:
                fault_cfg_i = _build_fault_cfg(cfg, args.inject, i,
                                               eol_map.get(i), _n_win, _spw)

            df, sa, eol, failed, twin = simulate(params, sim_cfg, traj_rng, fault_cfg=fault_cfg_i)
            df.to_csv(out_dir / f"traj_{i:03d}.csv", index=False)
            params_list.append(params)
            g = f.create_group(f"traj_{i:03d}")
            for c in df.columns:
                g.create_dataset(c, data=df[c].values)
            g.create_dataset("subarray_features", data=sa.astype(np.float32))
            if twin is not None:                          # T1: 落盘孪生静态量 (twin_only 不进模型输入, M4 前置)
                for tname, tval in twin.items():
                    if tname.startswith("_"):
                        continue                           # __verify_* 为内部验证用, 不落盘
                    g.create_dataset(tname, data=tval.astype(np.float32) if tval.dtype == np.float64 else tval)
                g.attrs["label_level"] = "subarray_train_label"
            for k, v in params.items():
                g.attrs[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
            if fault_cfg_i is not None:                      # §5.7: 故障注入元数据
                g.attrs["fault_type"] = fault_cfg_i["type"]
                g.attrs["fault_start_win"] = fault_cfg_i["start_win"]
                if "sub_id" in fault_cfg_i:
                    g.attrs["fault_sub_id"] = fault_cfg_i["sub_id"]
                g.attrs["paired_nominal_traj_id"] = i
            h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
            summaries.append(dict(
                idx=i, failed=bool(failed), eol_idx=eol,
                damage_end=float(df["damage"].iloc[-1]),
                SLL_end=float(df["SLL_dB_true"].iloc[-1]),
                theta_end=float(abs(df["theta_err_deg_true"].iloc[-1])),
                M_link_min=float(df["M_link_dB_true"].min()),
                Ea=params["Ea_eV"], Tj=params["Tj_base_C"], duty=params["duty"],
                scan=params["scan_az_deg"]))

    (out_dir / "params.json").write_text(json.dumps(
        [{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
          for k, v in p.items()} for p in params_list], indent=2), encoding="utf-8")

    n_fail = sum(s["failed"] for s in summaries)
    print(f">> 生成 {n_traj} 条轨迹 -> {out_dir}  (序列长 {len(pd.read_csv(out_dir/'traj_000.csv'))})")
    print(f">> 失效轨迹 {n_fail}/{n_traj} ({100*n_fail/n_traj:.0f}%)")
    # 多维寿命触发来源 (GPT 十三: 验证 SLL/θ_err 是否真的触发)
    if n_fail:
        by_M = sum(1 for s in summaries if s["failed"] and s["M_link_min"] <= 0)
        by_SLL = sum(1 for s in summaries if s["failed"] and s["SLL_end"] > lim_SLL)
        print(f">> 失效来源 (轨迹可多源): M_link≤0 {by_M} / SLL>{lim_SLL}dB {by_SLL}")
    if args.hash:
        print(f">> repro hash (seed={seed}, n_traj={n_traj}): {h.hexdigest()[:16]}")


lim_SLL = -8.0  # service_limits.SLL_max_dB (报告用; 实际从 config 读)


if __name__ == "__main__":
    main()
