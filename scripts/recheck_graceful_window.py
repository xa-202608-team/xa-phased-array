"""scripts/recheck_graceful_window.py

修正优雅降级窗口的统计口径 (原跨数据集比 median 是误口径)。

重算 1: Δ_first = EOL_svc - min_s(EOL_ch) 逐轨迹配对
  - "通道 EOL≪服务 EOL" 的通道 EOL 指首个越规范子阵 (min), 非 16 子阵中位 (median)
  - 逐轨迹内比 (同一条轨迹的 EOL_svc vs 它自己的 min_s EOL_ch), 不跨数据集
  - 报 Δ_first 的 P5/P50/P95 + Δ_first>0 占比 + Δ_first/T 视界占比

重算 2: EOL_svc 时刻服务越限归因
  - 三判据 (M_link≤0 / SLL>SLL_max / |θ|>θ_max) 哪条先触发的分布
  - 此刻 dropout 通道数 k/256 (k<26=10% → 连续幅相退化主导; k≥26 → dropout 主导)

用法: python scripts/recheck_graceful_window.py --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                    # noqa: E402


def recheck(seed: int, sim_path: Path, ch_path: Path, cfg: dict):
    SLL_max = float(cfg["sim"]["service_limits"]["SLL_max_dB"])
    theta_max = float(cfg["sim"]["service_limits"]["theta_err_max_deg"])
    deltas = []                 # Δ_first = EOL_svc - min_s(EOL_ch)
    eol_svcs = []               # 配对该轨迹服务寿命 (Δ_first 分母, 非仿真视界 8 年人为截止)
    triggers = {"M_link": 0, "SLL": 0, "theta": 0, "multi": 0, "none": 0}
    ks = []                     # dropout 通道数 @ EOL_svc
    eol_ch_all = []             # 所有失效通道 EOL (供 min vs median 对比)
    eol_svc_list = []
    n_traj = n_failed_svc = 0
    T_ref = None
    with h5py.File(sim_path, "r") as fsim, h5py.File(ch_path, "r") as fch:
        for key in sorted(fsim.keys()):
            n_traj += 1
            g = fsim[key]
            label_fail = g["label_fail"][:]
            if T_ref is None:
                T_ref = len(label_fail)
            failed_svc = bool(label_fail.any())
            eol_svc = int(np.argmax(label_fail)) if failed_svc else len(label_fail) - 1
            # 重算 1: 逐轨迹 min_s(EOL_ch)
            ch_traj = fch[key]
            eol_chs = []
            for sub_key in sorted(k for k in ch_traj.keys() if k.startswith("sub_")):
                sub = ch_traj[sub_key]
                if bool(sub.attrs["event_observed"]):
                    eol_chs.append(int(sub.attrs["eol_idx"]))
            eol_ch_all.extend(eol_chs)
            if eol_chs and failed_svc:
                deltas.append(eol_svc - min(eol_chs))
                eol_svcs.append(eol_svc)
            if failed_svc:
                eol_svc_list.append(eol_svc)
            # 重算 2: EOL_svc 时刻归因
            if failed_svc:
                n_failed_svc += 1
                M = float(g["M_link_dB_true"][eol_svc])
                SLL = float(g["SLL_dB_true"][eol_svc])
                th = float(g["theta_err_deg_true"][eol_svc])
                k = float(g["k_failed"][eol_svc])
                ks.append(k)
                tM = M <= 0.0
                tS = SLL > SLL_max
                tT = abs(th) > theta_max
                n_trig = int(tM) + int(tS) + int(tT)
                if n_trig == 0:
                    triggers["none"] += 1
                elif n_trig == 1:
                    triggers["M_link" if tM else ("SLL" if tS else "theta")] += 1
                else:
                    triggers["multi"] += 1

    T = T_ref or 1
    print(f"\n{'='*64}")
    print(f"重算 (seed={seed}, n_traj={n_traj}, failed_svc={n_failed_svc}, T={T})")
    print(f"{'='*64}")
    # 口径对比: 旧 (跨数据集 median) vs 新 (逐轨迹 min)
    if eol_ch_all and eol_svc_list:
        print(f"\n[口径对比] 旧跨数据集 median: EOL_ch={np.median(eol_ch_all):.0f} vs "
              f"EOL_svc={np.median(eol_svc_list):.0f} "
              f"({'ch<svc' if np.median(eol_ch_all)<np.median(eol_svc_list) else 'ch>=svc'})")
    if deltas:
        d = np.array(deltas, dtype=float)
        esv = np.array(eol_svcs, dtype=float)
        win_h = 6.0   # sample_period_s=21600s = 6h/窗
        print(f"\n--- 重算 1: Δ_first = EOL_svc - min_s(EOL_ch)  (逐轨迹配对, n={len(d)}) ---")
        print(f"  Δ_first 分位: P5={np.percentile(d,5):.0f}  P50={np.percentile(d,50):.0f}  "
              f"P95={np.percentile(d,95):.0f} 窗  (1 窗=6h)")
        print(f"  Δ_first 绝对量: 中位 ≈ {np.median(d)*win_h/24:.1f} 天,  "
              f"P95 ≈ {np.percentile(d,95)*win_h/24/365:.2f} 年")
        print(f"  Δ_first > 0 占比: {np.mean(d>0)*100:.1f}%  (服务越限晚于首个器件越限)")
        n_neg = int(np.sum(d <= 0))
        print(f"  Δ_first ≤ 0: {n_neg}/{len(d)} ({100*n_neg/len(d):.1f}%) — "
              f"服务越限早于任何子阵越规范 (阵因子聚合先耗余量, 两层架构必要性论据)")
        # 分母 = 该轨迹自身服务寿命 (非仿真视界; 视界=8 年人为截止, 非物理量)
        frac = d / np.maximum(esv, 1.0)
        print(f"  Δ_first / EOL_svc 中位: {np.median(frac)*100:.1f}%  "
              f"(占该轨迹服务寿命; 10-20% → 优雅降级叙事成立)")
        print(f"  Δ_first / EOL_svc P95: {np.percentile(frac,95)*100:.1f}%")
    else:
        print("\n--- 重算 1: 无配对样本 (缺失效通道或失效轨迹) ---")
    print(f"\n--- 重算 2: EOL_svc 时刻服务越限归因 ---")
    total = sum(triggers.values())
    for k, v in triggers.items():
        print(f"  {k:>8}: {v:>4} ({100*v/max(total,1):.1f}%)")
    if ks:
        ks_arr = np.array(ks)
        print(f"\n  dropout 通道数 k/256 @ EOL_svc:")
        print(f"    中位={np.median(ks_arr):.0f}  P50={np.percentile(ks_arr,50):.0f}  "
              f"P95={np.percentile(ks_arr,95):.0f}  max={ks_arr.max():.0f}")
        print(f"    k<26 (10%): {np.mean(ks_arr<26)*100:.1f}%   k≥26 (10%): {np.mean(ks_arr>=26)*100:.1f}%")
        if np.median(ks_arr) < 26:
            print(f"    → k 中位 <10%: 服务越限由**连续幅相退化+AF恶化**主导, dropout 只是伴随现象")
            print(f"      (两阈值时间上碰巧靠近, 不构成建模缺陷; A 路线成立)")
        else:
            print(f"    → k 中位 ≥10%: dropout 确实主导服务越限 (需查是否与 z>=1 同事件 → B′)")
    print(f"\n{'='*64}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed
    sim_path = ROOT / f"data/simulated/phased_array/sim_v2/seed_{seed}/phased_array_all.h5"
    ch_path = ROOT / cfg["channel_level"]["feature_path"]
    if not sim_path.exists():
        print(f"!! 缺 {sim_path}; 先 python -m src.sim.phased_array_sim --seed {seed} --n_traj 50")
        sys.exit(1)
    if not ch_path.exists():
        print(f"!! 缺 {ch_path}; 先 python -m src.sim.build_channel_hi --config {args.config}")
        sys.exit(1)
    recheck(seed, sim_path, ch_path, cfg)


if __name__ == "__main__":
    main()
