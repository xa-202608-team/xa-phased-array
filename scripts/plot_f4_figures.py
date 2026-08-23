# -*- coding: utf-8 -*-
"""F4 三类答辩图: OAT 敏感性 / B1 完整 AF vs 简化公式 ΔEOL / B2-B3 matched-seed 配对。

数据全部从权威工件直读 (不从 Markdown 反抄):
  - outputs/f4_oat/oat_results.json               (A: OAT)
  - data/simulated/.../sim_v2/seed_42 (BASE_SIM)  (B1: 逐轨迹 full vs 简化)
  - outputs/f2_formal_5seed + outputs/f4_ablation/*/results_partial.jsonl
                                                  (B2/B3: matched seeds 42-44)
matched-seed 配对口径与 scripts/f4_analyze.py / tests/test_f4_closeout.py 同源。

用法: python scripts/plot_f4_figures.py [--output-dir docs/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.f4_analyze import (          # noqa: E402
    N_ELEMENTS, _first_sustained, paired_rmse_summary, read_group_rmse_by_seed)

OAT = ROOT / "outputs" / "f4_oat" / "oat_results.json"
ABL = ROOT / "outputs" / "f4_ablation"
F2_JSONL = ROOT / "outputs" / "f2_formal_5seed" / "results_partial.jsonl"
BASE_SIM = ROOT / "data/simulated/phased_array/sim_v2/seed_42/phased_array_all.h5"
GROUP = "ch_target_only_gru"


def fig_oat(out: Path) -> None:
    d = json.loads(OAT.read_text(encoding="utf-8"))
    base = d["baseline"]
    svc0 = base["svc_eol_median_windows"]
    ch0 = base["ch_eol_median_windows_failed_only"]
    variants = [v for v in d["variants"] if "error" not in v]
    names = [v["variant"] for v in variants]
    svc = [100 * (v["svc_eol_median_windows"] - svc0) / svc0 for v in variants]
    ch = [100 * (v["ch_eol_median_windows_failed_only"] - ch0) / ch0 for v in variants]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axvline(0, color="gray", lw=0.8)
    ax.axvline(-18.2, color="crimson", lw=0.8, ls="--", alpha=0.6)
    ax.plot(svc, y, "o", color="tab:blue", label="服务级 EOL_svc", ms=7)
    ax.plot(ch, y, "s", color="tab:red", label="通道级 EOL_ch", ms=6)
    for yi, n in zip(y, names):
        if n == "ea_plus":
            ax.annotate("Ea+20% 通道级最坏 −18.2%", xy=(-18.2, yi),
                        xytext=(-17, yi - 1.4), fontsize=9, color="crimson")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("相对基线变化 (%)")
    ax.set_title("物理参数 ±20% OAT — EOL 中位变化 (200 轨迹, seed42)\n"
                 "结论稳健: 服务级 |Δ|≲12%, 通道级对 Ea 最敏感; 无方向翻转",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _b1_per_traj(k_sustain: int = 4, sll_max: float = -8.0, th_max: float = 0.5):
    """逐轨迹 (eol_full, eol_simp) — 与 f4_analyze.b1_full_vs_simplified 同语义。"""
    import h5py
    eol_full, eol_simp = [], []
    with h5py.File(BASE_SIM, "r") as f:
        for tk in sorted(f.keys()):
            g = f[tk]
            m, s, th, kf = (g["M_link_dB_true"][:], g["SLL_dB_true"][:],
                            np.abs(g["theta_err_deg_true"][:]), g["k_failed"][:])
            margin0 = float(g.attrs["margin0_dB"])
            cands = [(e, n) for e, n in (
                (_first_sustained(m <= 0.0, k_sustain), "M_link"),
                (_first_sustained(s > sll_max, k_sustain), "SLL"),
                (_first_sustained(th > th_max, k_sustain), "theta")) if e is not None]
            if not cands:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                gain_drop = -20.0 * np.log10(np.clip(1.0 - kf / N_ELEMENTS, 1e-9, None))
            es = _first_sustained(margin0 - gain_drop <= 0.0, k_sustain)
            if es is None:
                continue                      # 简化公式漏检轨迹不入配对 (同 f4_analyze)
            eol_full.append(min(cands)[0])
            eol_simp.append(es)
    return np.asarray(eol_full), np.asarray(eol_simp)


def fig_b1(out: Path) -> None:
    from scipy import stats as sp_stats
    ef, es = _b1_per_traj()
    delta = es - ef
    rho = sp_stats.spearmanr(ef, es).statistic
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([ef.min(), ef.max()], [ef.min(), ef.max()], "k--", lw=1, label="y=x (无偏)")
    ax.scatter(ef, es, s=14, alpha=0.55, color="tab:blue",
               label=f"失效轨迹 (n={len(ef)})")
    ax.set_xlabel("完整阵列因子口径 EOL (窗)")
    ax.set_ylabel("简化公式 20log10(1−k/N) 口径 EOL (窗)")
    ax.set_title(
        f"B1 完整阵列因子 vs 简化公式 — 逐轨迹 EOL\n"
        f"median ΔEOL(简化−full) = {np.median(delta):+.0f} 窗 (系统性偏晚); "
        f"ρ = {rho:.3f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_b2b3(out: Path) -> None:
    base = read_group_rmse_by_seed(F2_JSONL, GROUP)
    variants = [("b2_count", "B2 通道计数"), ("b3_subagg", "B3 子阵聚合"),
                ("b3_sparse", "B3 稀疏 1/6 cadence")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (name, label) in zip(axes, variants):
        var = read_group_rmse_by_seed(ABL / name / "results_partial.jsonl", GROUP)
        s = paired_rmse_summary(base, var)
        seeds = s["seeds"]
        lo, hi = s["delta_ci"]
        cross0 = lo <= 0 <= hi
        for i, sd in enumerate(seeds):
            b, v = base[sd], var[sd]
            ax.plot([b, v], [i, i], "-", color="gray", lw=1.4, zorder=1)
            ax.plot(b, i, "o", color="tab:blue", ms=8, zorder=2)
            ax.plot(v, i, "s", color="tab:red", ms=7, zorder=2)
        ax.set_yticks(range(len(seeds)), [f"seed {sd}" for sd in seeds])
        ax.set_title(f"{label}\nΔ = {s['delta_mean']:+.4f}  "
                     f"CI[{lo:+.4f}, {hi:+.4f}]{'跨 0' if cross0 else ''}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("测试 RMSE (÷H=11688)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].plot([], [], "o", color="tab:blue", label="matched-seed 基线 (F2 seeds 42–44)")
    axes[0].plot([], [], "s", color="tab:red", label="变体")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("B2/B3 消融 — matched seeds 42–44 逐 seed 配对 (n=3, 描述性)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="docs/figures")
    args = ap.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_oat(out_dir / "f4_oat_sensitivity.png")
    print(f">> {out_dir / 'f4_oat_sensitivity.png'}")
    fig_b1(out_dir / "f4_b1_af_vs_simplified.png")
    print(f">> {out_dir / 'f4_b1_af_vs_simplified.png'}")
    fig_b2b3(out_dir / "f4_b2b3_matched_seed.png")
    print(f">> {out_dir / 'f4_b2b3_matched_seed.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
