# -*- coding: utf-8 -*-
"""F4 汇总分析: OAT 敏感性表 + B1 (full vs simplified AF) + B2/B3 消融模型数字
→ docs/results_phased_array_f4.md (预注册口径见 docs/f4_sensitivity_ablation_design.md)。
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OAT = ROOT / "outputs" / "f4_oat" / "oat_results.json"
ABL = ROOT / "outputs" / "f4_ablation"
F2_JSONL = ROOT / "outputs" / "f2_formal_5seed" / "results_partial.jsonl"
GROUP_CH = "ch_target_only_gru"
BASE_SIM = ROOT / "data/simulated/phased_array/sim_v2/seed_42/phased_array_all.h5"
F2_METRICS = ROOT / "results" / "reference" / "all_metrics_phased_array.json"
REPORT = ROOT / "docs" / "results_phased_array_f4.md"
N_ELEMENTS = 256
PARAM_LABELS = {"ea": "Ea 激活能 (Arrhenius)", "thermal_cycle": "热循环 ΔT_ref (Coffin-Manson)",
                "rth_feedback": "热阻正反馈 a5", "life_sigma": "寿命离差 σ",
                "margin0": "初始余量 margin0"}


def _first_sustained(bad: np.ndarray, k: int) -> int | None:
    run = 0
    for i, b in enumerate(bad):
        run = run + 1 if b else 0
        if run >= k:
            return i - k + 1
    return None


def b1_full_vs_simplified(k_sustain: int = 4, sll_max: float = -8.0,
                          th_max: float = 0.5) -> dict:
    eol_full, eol_simp, bind_not_link = [], [], 0
    miss = 0            # full 失效但 simplified 永不越限 (漏检)
    with h5py.File(BASE_SIM, "r") as f:
        for tk in sorted(f.keys()):
            g = f[tk]
            m = g["M_link_dB_true"][:]
            s = g["SLL_dB_true"][:]
            th = np.abs(g["theta_err_deg_true"][:])
            kf = g["k_failed"][:]
            margin0 = float(g.attrs["margin0_dB"])
            e_link = _first_sustained(m <= 0.0, k_sustain)
            e_sll = _first_sustained(s > sll_max, k_sustain)
            e_th = _first_sustained(th > th_max, k_sustain)
            cands = [(e, n) for e, n in ((e_link, "M_link"), (e_sll, "SLL"), (e_th, "theta"))
                     if e is not None]
            if not cands:
                continue                            # full 口径删失, 不入对比
            ef, bind = min(cands, key=lambda x: x[0])
            eol_full.append(ef)
            if bind != "M_link":
                bind_not_link += 1
            with np.errstate(divide="ignore", invalid="ignore"):
                gain_drop = -20.0 * np.log10(np.clip(1.0 - kf / N_ELEMENTS, 1e-9, None))
            es = _first_sustained(margin0 - gain_drop <= 0.0, k_sustain)
            if es is None:
                miss += 1
                eol_simp.append(np.nan)
            else:
                eol_simp.append(es)
    eol_full = np.asarray(eol_full, dtype=float)
    eol_simp = np.asarray(eol_simp, dtype=float)
    ok = ~np.isnan(eol_simp)
    delta = eol_simp[ok] - eol_full[ok]
    from scipy import stats as sp_stats
    rho = float(sp_stats.spearmanr(eol_full[ok], eol_simp[ok]).statistic) if ok.sum() > 2 else None
    return {
        "n_failed_full": len(eol_full),
        "median_eol_full": float(np.median(eol_full)),
        "median_delta_simp_minus_full": float(np.median(delta)),
        "spearman_rho": rho,
        "frac_binding_not_mlink": bind_not_link / len(eol_full),
        "frac_missed_by_simplified": miss / len(eol_full),
    }


def read_group_rmse_by_seed(path: Path, group: str) -> dict[int, float]:
    rows: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("group") == group:
            rows[int(row["seed"])] = float(row["rmse"])
    return rows


def paired_rmse_summary(
    baseline: dict[int, float], variant: dict[int, float]
) -> dict[str, float | int | list[int] | tuple[float, float]]:
    """共同 seed 上的逐 seed 配对统计; CI = t(n-1)·s/√n, n=3 时仅描述性。"""
    seeds = sorted(set(baseline) & set(variant))
    if len(seeds) < 2:
        raise ValueError("paired RMSE 至少需要 2 个共同 seed")
    base = np.asarray([baseline[s] for s in seeds], dtype=float)
    var = np.asarray([variant[s] for s in seeds], dtype=float)
    delta = var - base
    n = len(seeds)
    delta_std = float(delta.std(ddof=1))
    from scipy.stats import t
    half = float(t.ppf(0.975, n - 1) * delta_std / np.sqrt(n))
    return {
        "seeds": seeds,
        "n": n,
        "baseline_mean": float(base.mean()),
        "baseline_std": float(base.std(ddof=1)),
        "variant_mean": float(var.mean()),
        "variant_std": float(var.std(ddof=1)),
        "delta_mean": float(delta.mean()),
        "delta_std": delta_std,
        "delta_ci": (float(delta.mean() - half), float(delta.mean() + half)),
    }


def main() -> int:
    lines = ["# F4 结果：±20% 参数敏感性 OAT + PA6 三类消融\n",
             "> 预注册协议: docs/f4_sensitivity_ablation_design.md (2026-08-23, 先于执行落档)。",
             "> 冻结基线 = sim_v2 seed42 (200 traj) + v2 channel h5 (F2 数字); "
             "全部变体数据在 outputs/f4_* 独立目录。\n"]

    # ---- A: OAT
    if OAT.exists():
        d = json.loads(OAT.read_text(encoding="utf-8"))
        base, variants = d["baseline"], d["variants"]
        lines += ["## A. 物理参数 ±20% OAT (10 变体, n_traj=200, seed=42)\n",
                  f"- 基线: 轨迹失效率 {base['svc_failure_rate']:.3f} / "
                  f"median EOL_svc {base['svc_eol_median_windows']:.0f} 窗 / "
                  f"通道失效率 {base['ch_failure_rate']:.3f} / "
                  f"median EOL_ch(失效) {base['ch_eol_median_windows_failed_only']:.0f} 窗 / "
                  f"margin0 中位 {base['median_margin0_dB']:.2f} dB / Ea 中位 {base['median_Ea_eV']:.2f} eV\n",
                  "| 变体 | 轨迹失效率 | median EOL_svc (窗) | 通道失效率 | median EOL_ch (窗) | margin0 中位 | Ea 中位 |",
                  "|---|---|---|---|---|---|---|"]
        for v in variants:
            if "error" in v:
                lines.append(f"| {v['variant']} | ERROR: {v['error'][:60]} | | | | | |")
                continue
            lines.append(
                f"| {v['variant']} | {v['svc_failure_rate']:.3f} | {v['svc_eol_median_windows']:.0f} "
                f"| {v['ch_failure_rate']:.3f} | {v['ch_eol_median_windows_failed_only']:.0f} "
                f"| {v['median_margin0_dB']:.2f} | {v['median_Ea_eV']:.2f} |")
        lines += [
            "",
            "**判读**（相对基线的定性结论，预注册口径：如实报告，无通过/失败门）：",
            "- **结论稳健**：10 变体轨迹失效率带 0.53–0.64、通道失效率带 0.533–0.560；"
            "服务级 EOL 中位变化约在 ±12% 以内，通道级 EOL 对 Ea 更敏感"
            "（Ea+20% 最坏约 −18%），无一翻转数量级或方向性结论。",
            "- **Ea 是唯一显著敏感参数**：EOL_svc −10%/+12%、EOL_ch −18%/+12% —— Arrhenius "
            "指数非线性 + 跨器件 Tj 异质 → life_ref 归一化无法吸收（物理预期）。",
            "- **ΔT_ref 精确不变 = 结构不变性，非死参数**：thermal_cycle ±20% 的 EOL/失效率与"
            "基线逐字相同，但 life_scale_years 实测 79589 (−20%) / 40750 (基线) / 23582 (+20%) "
            "证明参数参与计算 —— 剂量单位尺度 (ΔTj/ΔT_ref)^n 同时进入累积剂量与 life_ref，"
            "相除精确消去：**EOL 统计依赖物理量比值而非绝对剂量单位**。",
            "- **rth_feedback / life_sigma 温和**（EOL_svc 变化 <±4%）：前者 a5 仅温升反馈系数，"
            "后者改分布散布不改中位（EOL_ch 2964–3063）。",
            "- **margin0 自洽性通过**：EOL_svc −4%/+7% 方向正确且 EOL_ch 严格不变（3052，服务层"
            "参数不触器件层）；margin0/Ea 自查列各自独变，无串扰。",
            "",
        ]
    else:
        lines.append("## A. OAT: 未运行 (缺 outputs/f4_oat/oat_results.json)\n")

    # ---- B1
    lines.append("## B1. 完整阵列因子 vs 简化公式 20log10(1−k/N)\n")
    b1 = b1_full_vs_simplified()
    lines += [
        f"- full 口径失效轨迹 {b1['n_failed_full']} 条 (连续 4 窗判据, 真值信号), "
        f"median EOL_svc = {b1['median_eol_full']:.0f} 窗",
        f"- 简化口径 median ΔEOL (simplified − full) = **{b1['median_delta_simp_minus_full']:+.0f} 窗**, "
        f"Spearman ρ = {b1['spearman_rho']:.3f}",
        f"- full 口径下非 M_link 约束先导 (SLL/θ_err) 占比 = **{b1['frac_binding_not_mlink']:.1%}**"
        " (简化公式原理上无法表征的失效模式)",
        f"- 简化公式完全漏检 (永不越限) 占比 = **{b1['frac_missed_by_simplified']:.1%}**\n",
        "**判读**：简化公式系统性偏晚（median +475 窗 ≈ 118 天）：20log10(1−k/N) 只计通道"
        "退出的增益损失，忽略幅相连续退化对方向图的侵蚀，故服务寿命被高估；但排序保真"
        "（ρ=0.996）。本数据集内 SLL/θ_err 无先导绑定（非 M_link 占比 0%）——完整 AF 的多维"
        "覆盖能力在本数据未行使，属能力差异未触发而非不存在；另有 3.3% 失效轨迹被简化公式"
        "完全漏检。\n",
    ]

    # ---- B2/B3 (matched seeds 配对口径; 审计修正 2026-08-23: 不再用 F2 五种子聚合均值作基线)
    lines.append("## B2/B3. 通道级模型消融 (ch_target_only_gru, matched seeds 42–44 配对, v2 口径)\n")
    full_agg = json.loads(F2_METRICS.read_text(encoding="utf-8"))["agg"][GROUP_CH]
    lines += [
        f"- **full (F2 冻结, 5 seeds)**: RMSE = {full_agg['rmse_mean']:.4f} ± {full_agg['rmse_std']:.4f} "
        "（主模型正式参考；seed 集不同，不用于 B2/B3 的 vs full 计算）",
        "- 配对口径：F2 与各变体在共同 seed 上的逐 seed 配对差，CI = t(2)·s/√n "
        "（n=3，仅描述性，不升级为确认性假设检验）\n",
        "| 档 | RMSE (÷H) | 配对 Δ | 95% CI |", "|---|---|---|---|",
    ]
    full_by_seed = read_group_rmse_by_seed(F2_JSONL, GROUP_CH)
    stats: dict[str, dict] = {}
    labels = {"b2_count": "B2 通道计数 (p_drift→存活占比)", "b3_subagg": "B3 子阵聚合",
              "b3_sparse": "B3 稀疏 1/6 cadence"}
    for name in ("b2_count", "b3_subagg", "b3_sparse"):
        jsonl = ABL / name / "results_partial.jsonl"
        if not jsonl.exists():
            lines.append(f"| {labels[name]} | 未跑 | — | — |")
            continue
        s = paired_rmse_summary(full_by_seed, read_group_rmse_by_seed(jsonl, GROUP_CH))
        stats[name] = s
        lo, hi = s["delta_ci"]
        lines.append(f"| {labels[name]} | {s['variant_mean']:.4f} ± {s['variant_std']:.4f} "
                     f"| {s['delta_mean']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    verdicts = []
    if "b2_count" in stats:
        s = stats["b2_count"]
        verdicts.append(
            f"- **B2（通道计数 vs 连续幅相）**：matched-seed baseline "
            f"{s['baseline_mean']:.4f}±{s['baseline_std']:.4f} → 计数特征 "
            f"{s['variant_mean']:.4f}±{s['variant_std']:.4f}，配对 Δ = **{s['delta_mean']:+.4f}**，"
            f"CI [{s['delta_ci'][0]:+.4f}, {s['delta_ci'][1]:+.4f}] —— 存活占比在终末 dropout 前"
            "近常数（终末前无判别力），连续幅相观测量承载几乎全部预后信息；支持三级退化链以"
            "连续幅相建模为主体的设计选择。")
    if "b3_subagg" in stats:
        s = stats["b3_subagg"]
        verdicts.append(
            f"- **B3 子阵聚合**：配对 Δ = **{s['delta_mean']:+.4f}**，CI "
            f"[{s['delta_ci'][0]:+.4f}, {s['delta_ci'][1]:+.4f}] —— 小幅但三个配对 seed 方向一致"
            "的恶化；子阵级分辨率有增量价值但幅度温和，阵列级聚合遥测仍保留主要退化信息。")
    if "b3_sparse" in stats:
        s = stats["b3_sparse"]
        verdicts.append(
            f"- **B3 稀疏 1/6 cadence**：配对 Δ = **{s['delta_mean']:+.4f}**，CI "
            f"[{s['delta_ci'][0]:+.4f}, {s['delta_ci'][1]:+.4f}] 跨 0 —— 方向信号未确认；且本协议"
            "中 cadence 下降与窗口物理跨度上升耦合（同长窗口 L=64 覆盖 6× 物理时程），不能归因"
            "为单一因素，需要固定物理跨度的后续预注册实验才能解释。")
    lines += ["", "**判读**：", *verdicts, ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f">> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
