#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/run_fault_eval.py — 故障注入 §5.7 配对评估。

三类输出:
  (a) 预测器降级: 冻结模型 (如有 ckpt) 或跳过; 当前版本先做 (b)(c)
  (b) 故障经三级链传播: 配对 ΔEOL_svc + 首绑定类型 + 瓶颈子阵命中率 (走真值, 不需模型)
  (c) G1-G5 验收门检查

用法:
    python scripts/run_fault_eval.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM_V2 = ROOT / "data/simulated/phased_array/sim_v2/seed_42/phased_array_all.h5"
FAULT_DIRS = {
    "rth_step":      ROOT / "data/simulated/phased_array/fault_rth_step/seed_42/phased_array_all.h5",
    "thermal_bias":  ROOT / "data/simulated/phased_array/fault_thermal_bias/seed_42/phased_array_all.h5",
    "channel_open":  ROOT / "data/simulated/phased_array/fault_channel_open/seed_42/phased_array_all.h5",
    "cal_freeze":    ROOT / "data/simulated/phased_array/fault_cal_freeze/seed_42/phased_array_all.h5",
}
N_FAULT = 15
CONSEC = 4
SLL_MAX = -8.0
THETA_MAX = 0.5


def _compute_eol_and_binding(g):
    """从 h5 group 算服务 EOL 索引 + 首绑定类型 + 瓶颈子阵。"""
    ml = g["M_link_dB_true"][:]
    sll = g["SLL_dB_true"][:]
    th = g["theta_err_deg_true"][:]
    violate_ml = ml <= 0
    violate_sll = sll > SLL_MAX
    violate_th = np.abs(th) > THETA_MAX
    violate = violate_ml | violate_sll | violate_th
    cs = np.convolve(violate.astype(int), np.ones(CONSEC, dtype=int), mode="full")[:len(violate)]
    if not (cs >= CONSEC).any():
        return None, "censored", None
    eol = int(np.argmax(cs >= CONSEC))
    # 首绑定: eol 时刻哪个先触发
    bindings = []
    if violate_ml[eol]: bindings.append("M_link")
    if violate_sll[eol]: bindings.append("SLL")
    if violate_th[eol]: bindings.append("theta_err")
    primary = bindings[0] if bindings else "M_link"
    # 瓶颈子阵: eol 时刻损伤最大的子阵
    bottleneck = None
    if "latent_sub_damage" in g:
        sd = g["latent_sub_damage"][:]  # (T, 16)
        bottleneck = int(np.argmax(sd[eol]))
    return eol, primary, bottleneck


def _compute_device_eol(g):
    """算首个器件越限时刻 (z>=1 的子阵)。"""
    if "latent_sub_damage" not in g:
        return None
    sd = g["latent_sub_damage"][:]  # (T, 16)
    exceed = sd >= 1.0
    if not exceed.any():
        return None
    return int(np.argmax(exceed.any(axis=1)))


def evaluate():
    results = {}
    # 标称
    nominal_eols = {}
    nominal_bindings = {}
    nominal_dev_eols = {}
    nominal_bottlenecks = {}
    with h5py.File(SIM_V2, "r") as f:
        for i in range(N_FAULT):
            g = f[f"traj_{i:03d}"]
            eol, bind, bn = _compute_eol_and_binding(g)
            nominal_eols[i] = eol
            nominal_bindings[i] = bind
            nominal_dev_eols[i] = _compute_device_eol(g)
            if bn is not None:
                nominal_bottlenecks[i] = bn

    print(f"标称: 失效 {sum(1 for v in nominal_eols.values() if v)}/{N_FAULT}")
    print(f"  首绑定: { {b: sum(1 for v in nominal_bindings.values() if v==b) for b in set(nominal_bindings.values())} }")
    print()

    for ft_name, h5_path in FAULT_DIRS.items():
        if not h5_path.exists():
            print(f"[跳过] {ft_name}: {h5_path} 不存在")
            continue
        rows = []
        with h5py.File(h5_path, "r") as f:
            for i in range(N_FAULT):
                g = f[f"traj_{i:03d}"]
                eol_f, bind_f, bn_f = _compute_eol_and_binding(g)
                dev_eol_f = _compute_device_eol(g)
                eol_n = nominal_eols.get(i)
                # 配对 ΔEOL_svc (步 → 天: 1步=6h, 4步=1天)
                delta_eol = None
                if eol_f is not None and eol_n is not None:
                    delta_eol = (eol_n - eol_f) * 6 / 24  # 天 (正=故障提前)
                elif eol_f is not None and eol_n is None:
                    delta_eol = -(eol_f * 6 / 24)  # 标称删失但故障失效 → 负值
                # 瓶颈子阵命中 (F1/F3 有注入子阵)
                hit = None
                if bn_f is not None:
                    injected_sub = g.attrs.get("fault_sub_id", None)
                    if injected_sub is not None:
                        hit = int(bn_f) == int(injected_sub)
                rows.append(dict(
                    traj=i, eol_fault=eol_f, eol_nominal=eol_n,
                    delta_eol_days=delta_eol,
                    binding_fault=bind_f, binding_nominal=nominal_bindings.get(i),
                    bottleneck_fault=bn_f, injected_sub=int(g.attrs.get("fault_sub_id", -1)),
                    bottleneck_hit=hit,
                    device_eol_fault=dev_eol_f, device_eol_nominal=nominal_dev_eols.get(i),
                ))

        # 汇总
        deltas = [r["delta_eol_days"] for r in rows if r["delta_eol_days"] is not None]
        bindings = [r["binding_fault"] for r in rows]
        from collections import Counter
        bind_counts = Counter(bindings)
        hits = [r["bottleneck_hit"] for r in rows if r["bottleneck_hit"] is not None]
        n_fault_fail = sum(1 for r in rows if r["eol_fault"] is not None)
        n_nom_fail = sum(1 for r in rows if r["eol_nominal"] is not None)

        summary = dict(
            type=ft_name,
            n=N_FAULT,
            n_fault_failed=n_fault_fail,
            n_nominal_failed=n_nom_fail,
            delta_eol_days_median=float(np.median(deltas)) if deltas else None,
            delta_eol_days_mean=float(np.mean(deltas)) if deltas else None,
            delta_eol_days_p25=float(np.percentile(deltas, 25)) if deltas else None,
            delta_eol_days_p75=float(np.percentile(deltas, 75)) if deltas else None,
            binding_counts=dict(bind_counts),
            bottleneck_hit_rate=float(np.mean(hits)) if hits else None,
            bottleneck_n=len(hits),
            rows=rows,
        )
        results[ft_name] = summary

        # 打印
        print(f"=== {ft_name} ===")
        print(f"  失效: 标称 {n_nom_fail}/{N_FAULT} → 故障 {n_fault_fail}/{N_FAULT}")
        if deltas:
            print(f"  ΔEOL_svc (天): 中位 {np.median(deltas):.1f} / 均值 {np.mean(deltas):.1f} "
                  f"/ P25-P75 [{np.percentile(deltas,25):.1f}, {np.percentile(deltas,75):.1f}]")
        print(f"  首绑定: {dict(bind_counts)}")
        if hits:
            print(f"  瓶颈子阵命中率: {np.mean(hits)*100:.1f}% ({sum(hits)}/{len(hits)}) vs 随机 6.25%")
        print()

    # 验收门
    print("=== 验收门 ===")
    for ft_name, s in results.items():
        deltas_abs = [abs(d) for d in [r["delta_eol_days"] for r in s["rows"] if r["delta_eol_days"] is not None]]
        g1 = "pass" if deltas_abs and np.median(deltas_abs) > 0 else "check"
        g2 = "pass" if s["delta_eol_days_median"] and s["delta_eol_days_median"] > 0 else "check"
        g4_bind = s["binding_counts"]
        sll_or_theta = g4_bind.get("SLL", 0) + g4_bind.get("theta_err", 0)
        g4 = "pass" if sll_or_theta / N_FAULT > 0.20 else f"fail ({sll_or_theta}/{N_FAULT}={sll_or_theta/N_FAULT*100:.0f}%)"
        print(f"  {ft_name}: G2(ΔEOL>0)={g2} / G4(SLL+θ>20%)={g4}")

    # 保存
    out = ROOT / "docs/实验结果汇总/11_故障注入/results_fault_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out}")
    return results


if __name__ == "__main__":
    evaluate()
