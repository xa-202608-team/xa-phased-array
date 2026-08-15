#!/usr/bin/env python
"""scripts/plot_physics_visualization.py — 相控阵物理可视化（方向图退化 + 热点场）

填补交付包缺少的两张关键物理图 (评审建议: 把最强的物理建模能力可视化):

  1. pa_pattern_degradation.png — t=0 / t=mid / t=EOL 的阵列方位切面方向图对比。
     展示主瓣增益下降、旁瓣抬升、波束指向漂移 — 三级退化链终点的直观可视化。

  2. pa_hotspot_field.png — 16×16 阵面二维热点温度场 + 同时刻通道退化 z 分布。
     展示失效空间聚簇 + 瓶颈子阵定位 — 支撑 §3.2.3 热点场 + 两层架构论点。

数据来源: 用 config 参数 + fixed seed 重新仿真一条代表性失效轨迹,
从 _simulate_subdose 返回的 twin 静态量重建任意时刻的通道幅相 → 阵列方向图。
不修改仿真器代码, 仅复用其物理方程 (src.sim.phased_array_sim)。

用法:
  python scripts/plot_physics_visualization.py [--config configs/phased_array.yaml]
                                               [--output-dir docs/figures]
                                               [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed                                   # noqa: E402
from src.sim.phased_array_sim import (                                        # noqa: E402
    sample_params, simulate, _array_pattern, _grid_positions,
    THETA_COARSE_DEG, SEC_PER_YEAR, K_B)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


# ===================================================================== 重建方向图
def reconstruct_array_pattern(twin, params, sim_cfg, f_sub_row, f_scalar_val):
    """从 twin 静态量重建给定时刻的阵列方向图。

    物理方程与 _simulate_subdose 第二级 + 第三级逐字一致:
      f_ch = f_sub[:, sa_ids] * c_elem
      RDS/IDSS/gm → P_ratio → a (幅度)
      dphi = deg2rad(dphi_max) * f_ch * eta_phi + grad_amp * f * (grad·pos)
      dropout: f_ch > dropout_thr → a=0
      patt = |Σ a_i exp(j dphi_i) exp(j 2π (u-u0) x_i)|²
    """
    grid = sim_cfg["array"]["grid"]
    d_lambda = float(sim_cfg["array"]["element_spacing_lambda"])
    n_ch = int(sim_cfg["array"]["n_elements"])
    pos2d = _grid_positions(grid, d_lambda)
    pos_x = pos2d[:, 0]

    sa_ids = twin["twin_subarray_ids"]
    c_elem = twin["twin_c_elem"]
    eta_R = twin["twin_eta_R"]
    eta_phi = twin["twin_eta_phi"]
    grad_dir = twin["twin_grad_dir"]
    dropout_thr = twin["twin_dropout_thr"]

    # 第二级: f_ch → 器件参数漂移 → 通道幅相
    f_ch = np.clip(f_sub_row[sa_ids] * c_elem, 0, None)                  # (n_ch,)
    Delta_R = params["Delta_R"]
    RDS_ratio = 1.0 + (Delta_R - 1.0) * f_ch
    IDSS_ratio = np.clip(1.0 - params["decay_I"] * f_ch, 0.1, 1.0)
    gm_ratio = np.clip(1.0 - params["decay_g"] * f_ch, 0.1, 1.0)
    P_ratio = IDSS_ratio * gm_ratio / RDS_ratio
    a = np.sqrt(np.clip(P_ratio, 0.0, None)) * eta_R                     # (n_ch,)
    grad_amp = np.deg2rad(params["dphi_max_deg"]) * 0.3
    dphi_grad = grad_amp * f_scalar_val * (
        grad_dir[0] * pos2d[:, 0] + grad_dir[1] * pos2d[:, 1])
    dphi_rad = np.deg2rad(params["dphi_max_deg"]) * f_ch * eta_phi + dphi_grad
    # dropout (Weibull 退出)
    dropout_mask = f_ch > dropout_thr
    a = np.where(dropout_mask, 0.0, a)

    # 第三级: 阵列方向图
    u0 = np.sin(np.deg2rad(params["scan_az_deg"]))
    u_grid = np.sin(np.deg2rad(THETA_COARSE_DEG))
    patt = _array_pattern(a, dphi_rad, pos_x, u_grid, u0)               # (n_theta,)
    return patt, a, dropout_mask, f_ch


# ===================================================================== 重建热点场
def reconstruct_hotspot_field(params, sim_cfg, seed_traj):
    """用独立 traj_rng (同 seed) 按 _simulate_subdose 的 RNG 消耗顺序重建热点场。

    返回 pos2d, Tj_offset_elem (n_ch,), hotspot_centers 列表。
    RNG 消耗顺序: eta_phi → eta_R → grad_dir → xi_elem → 热点场循环。
    """
    grid = sim_cfg["array"]["grid"]
    d_lambda = float(sim_cfg["array"]["element_spacing_lambda"])
    n_ch = int(sim_cfg["array"]["n_elements"])
    pos2d = _grid_positions(grid, d_lambda)
    p = sim_cfg["physics"]
    hs = p.get("hotspot_field", {})
    traj_rng = np.random.default_rng(seed_traj)

    # 与 _simulate_subdose 逐字一致的 RNG 消耗 (第575-592行)
    traj_rng.normal(0.0, 1.0, n_ch)        # eta_phi
    traj_rng.normal(1.0, 0.05, n_ch)       # eta_R
    traj_rng.normal(0.0, 1.0, 2)           # grad_dir
    traj_rng.normal(0.0, 1.0, n_ch)        # xi_elem

    # 热点场
    Tj_offset_elem = np.zeros(n_ch)
    centers = []
    if bool(hs.get("enabled", True)):
        n_hs = int(traj_rng.integers(*hs["n_hotspots_range"], endpoint=True))
        for _ in range(n_hs):
            cx = float(traj_rng.uniform(-1, 1)) * (grid[0] * d_lambda) * 0.5
            cy = float(traj_rng.uniform(-1, 1)) * (grid[1] * d_lambda) * 0.5
            rad = float(traj_rng.uniform(*hs["radius_lambda_range"])) * d_lambda
            boost_K = float(traj_rng.uniform(*hs["Tj_boost_C_range"]))
            d2 = np.sqrt((pos2d[:, 0] - cx) ** 2 + (pos2d[:, 1] - cy) ** 2)
            Tj_offset_elem += boost_K * np.exp(-0.5 * (d2 / max(rad, 1e-6)) ** 2)
            centers.append(dict(cx=cx, cy=cy, radius=rad, amplitude=boost_K))
    return pos2d, Tj_offset_elem, centers


# ===================================================================== 图1: 方向图退化
def plot_pattern_degradation(twin, params, sim_cfg, df, eol_idx, out_path):
    """阵列方位切面方向图退化对比 (t=0 / t=mid / t=EOL)。"""
    f_sub = twin["latent_sub_damage"]
    f_scalar = df["damage"].values
    n_out = len(f_scalar)
    mid_idx = eol_idx // 2

    theta = THETA_COARSE_DEG
    colors = ["#2563eb", "#f59e0b", "#dc2626"]
    labels_t = []

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, t_idx in enumerate([0, mid_idx, eol_idx]):
        patt, a, dropout, f_ch = reconstruct_array_pattern(
            twin, params, sim_cfg, f_sub[t_idx], f_scalar[t_idx])
        # 归一化到 t=0 峰值 (跨时刻可比的 dB 方向图)
        patt_db = 10.0 * np.log10(patt / (patt.max() + 1e-30) + 1e-30)
        years = df["t"].iloc[t_idx] / SEC_PER_YEAR
        n_drop = int(dropout.sum())
        label = f"t={years:.1f} 年" + (f" ({n_drop} 通道退出)" if n_drop > 0 else "")
        ax.plot(theta, patt_db, color=colors[idx], linewidth=1.8, alpha=0.85,
                label=label, zorder=5 - idx)

    # 服务越限阈值标注
    sll_max = float(sim_cfg["service_limits"]["SLL_max_dB"])
    ax.axhline(y=sll_max, color="#94a3b8", linestyle=":", linewidth=1.2,
               alpha=0.7, label=f"SLL 越限 ({sll_max:.0f} dB)")

    # 扫描角标注
    scan_az = params["scan_az_deg"]
    ax.axvline(x=scan_az, color="#16a34a", linestyle="--", linewidth=1, alpha=0.4)

    ax.set_xlabel("方位角 (°)", fontsize=12)
    ax.set_ylabel("归一化方向图 (dB)", fontsize=12)
    ax.set_ylim(-45, 5)
    ax.set_xlim(-60, 60)
    ax.set_title(f"阵列方向图退化 (轨迹扫描角 {scan_az:.1f}°, "
                 f"EOL={df['t'].iloc[eol_idx] / SEC_PER_YEAR:.1f} 年)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ===================================================================== 图2: 热点场 + z 分布
def plot_hotspot_field(twin, params, sim_cfg, df, eol_idx, pos2d, Tj_offset, centers, out_path):
    """16×16 阵面热点温度场 + EOL 时刻通道退化 z 分布。"""
    grid = sim_cfg["array"]["grid"]
    n_ch = int(sim_cfg["array"]["n_elements"])
    f_sub = twin["latent_sub_damage"]
    f_scalar = df["damage"].values

    # EOL 时刻通道级退化指标 z (用 delta_thresholds 归一)
    f_ch_eol = np.clip(f_sub[eol_idx][twin["twin_subarray_ids"]] * twin["twin_c_elem"], 0, None)
    delta_cfg = sim_cfg.get("channel_level", sim_cfg.get("physics", {}))
    dR = 0.35
    dI = 0.20
    dg = 0.15
    # z = max(dR_ratio, dI_ratio, dg_ratio) 近似 (用 f_ch 直接映射)
    z_R = (1.0 + (params["Delta_R"] - 1.0) * f_ch_eol - 1.0) / dR
    z_I = params["decay_I"] * f_ch_eol / dI
    z_g = params["decay_g"] * f_ch_eol / dg
    z_ch = np.maximum(np.maximum(z_R, z_I), z_g)

    Tj_grid = Tj_offset.reshape(grid[1], grid[0])  # reshape 到 16×16
    z_grid = z_ch.reshape(grid[1], grid[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左: 热点温度场
    ax = axes[0]
    im = ax.imshow(Tj_grid, cmap="hot", interpolation="bilinear", origin="lower")
    # 标注热点中心
    for c in centers:
        cx_idx = (c["cx"] / (grid[0] * float(sim_cfg["array"]["element_spacing_lambda"]))
                  + 0.5) * grid[0]
        cy_idx = (c["cy"] / (grid[1] * float(sim_cfg["array"]["element_spacing_lambda"]))
                  + 0.5) * grid[1]
        circle = plt.Circle((cx_idx, cy_idx), 2, fill=False, color="cyan",
                            linewidth=2, linestyle="--")
        ax.add_patch(circle)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("热点温升 ΔT_j (K)", fontsize=10)
    ax.set_title(f"二维热点温度场 ({len(centers)} 个热点)", fontsize=12, fontweight="bold")
    ax.set_xlabel("阵面 x (阵元索引)", fontsize=11)
    ax.set_ylabel("阵面 y (阵元索引)", fontsize=11)

    # 右: EOL 时刻通道 z 分布
    ax = axes[1]
    im2 = ax.imshow(z_grid, cmap="RdYlGn_r", interpolation="bilinear", origin="lower",
                    vmin=0, vmax=max(z_ch.max(), 1.5))
    # 标注失效通道 (z > 1)
    fail_mask = z_ch > 1.0
    if fail_mask.any():
        fail_idx = np.where(fail_mask)[0]
        for fi in fail_idx:
            fx = fi % grid[0]
            fy = fi // grid[0]
            ax.plot(fx, fy, "kx", markersize=8, markeredgewidth=1.5)
    cb2 = fig.colorbar(im2, ax=ax, pad=0.02)
    cb2.set_label("退化指标 z", fontsize=10)
    ax.set_title(f"EOL 时刻通道退化 z 分布 "
                 f"({int(fail_mask.sum())}/{n_ch} 越限)", fontsize=12, fontweight="bold")
    ax.set_xlabel("阵面 x (阵元索引)", fontsize=11)
    ax.set_ylabel("阵面 y (阵元索引)", fontsize=11)

    fig.suptitle(f"失效空间聚簇 — 轨迹 EOL={df['t'].iloc[eol_idx] / SEC_PER_YEAR:.1f} 年",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser(description="相控阵物理可视化")
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--output-dir", default="docs/figures")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(args.seed, cfg["reproducibility"]["deterministic"],
             cfg["reproducibility"]["cudnn_benchmark"])
    sim_cfg = dict(cfg["sim"])
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 采样代表性失效轨迹: 仿真前 30 条, 选失效轨迹中位寿命的
    rng = np.random.default_rng(args.seed)
    candidates = []
    for attempt in range(30):
        params = sample_params(rng, sim_cfg)
        traj_rng = np.random.default_rng(params["seed_traj"])
        df, sa, eol, failed, twin = simulate(params, sim_cfg, traj_rng)
        if failed and twin is not None:
            candidates.append((df, sa, eol, failed, twin, params))
    if not candidates:
        print(">> [warning] 30 条均未失效, 用最后一条")
        result = (df, sa, eol, failed, twin, params)
    else:
        candidates.sort(key=lambda x: x[0]["t"].iloc[x[2]] / SEC_PER_YEAR)
        result = candidates[len(candidates) // 2]
    df, sa, eol, failed, twin, params = result
    n_fail_total = len(candidates) if candidates else 0
    print(f">> 仿真 30 条 ({n_fail_total} 条失效), 选定中位寿命轨迹: "
          f"EOL={eol}, {df['t'].iloc[eol] / SEC_PER_YEAR:.1f} 年, "
          f"scan={params['scan_az_deg']:.1f}°")

    df, sa, eol, failed, twin, params = result

    # 重建热点场
    pos2d, Tj_offset, centers = reconstruct_hotspot_field(
        params, sim_cfg, params["seed_traj"])
    print(f">> 热点场: {len(centers)} 个热点, "
          f"max ΔTj={Tj_offset.max():.1f} K")

    print("=" * 60)
    print("1. 阵列方向图退化对比图")
    plot_pattern_degradation(twin, params, sim_cfg, df, eol,
                             out_dir / "pa_pattern_degradation.png")

    print("=" * 60)
    print("2. 热点温度场 + 通道 z 分布")
    plot_hotspot_field(twin, params, sim_cfg, df, eol,
                       pos2d, Tj_offset, centers,
                       out_dir / "pa_hotspot_field.png")

    print("=" * 60)
    print("完成! 图表保存到:", out_dir)


if __name__ == "__main__":
    main()
