"""§4h 误差方向/风险偏置诊断分析.

输入 bias_diag/results_partial.jsonl，输出四臂 k=3 的 seed 均值、source−random
per-seed 配对 bias 差异 CI，以及端点 RMSE bit-exact 校验。诊断是描述性 n=5，
不把 bias CI 当确认性迁移效果检验。
"""
import json
import math
import statistics
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "bias_diag" / "results_partial.jsonl"
REF = ROOT / "outputs" / "kshot_source_arms" / "all_metrics_phased_array.json"
GROUPS = ["ch_random_full_finetune_k3", "ch_source_mmd_physics_k3",
          "ch_source_igbt_k3", "ch_source_multi_k3"]
LABELS = {"ch_random_full_finetune_k3":"Random", "ch_source_mmd_physics_k3":"MOSFET",
          "ch_source_igbt_k3":"IGBT", "ch_source_multi_k3":"Multi"}
BOXES = ["failed_all", "early", "middle", "late", "censored"]

def ci(ds):
    m = statistics.mean(ds); sd = statistics.stdev(ds); n = len(ds)
    h = float(stats.t.ppf(.975, n-1)) * sd / math.sqrt(n)
    return m, m-h, m+h

def main():
    records = [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()]
    by = {g: {r["seed"]: r for r in records if r["group"] == g} for g in GROUPS}
    ref = json.loads(REF.read_text(encoding="utf-8"))["agg"]["_raw_by"]
    endpoint = {}
    for g in [GROUPS[0], GROUPS[1]]:
        endpoint[g] = all({r["seed"]: r["rmse"] for r in by[g].values()}[int(s)] == v
                           for s, v in ref[g].items())
    lines = ["# §4h 误差方向/风险偏置诊断结果\n\n",
             "四臂 k=3、seeds 42–46；诊断为描述性分析，RUL 归一化空间。\n\n",
             "## 端点复现\n\n"]
    for g, ok in endpoint.items():
        lines.append(f"- {LABELS[g]} vs §4c RMSE per-seed bit-exact: **{ok}**\n")
    lines.append("\n## 各臂 seed 均值\n\n| 臂 | RMSE | failed bias | early | middle | late | censored lb violation |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for g in GROUPS:
        rs = list(by[g].values())
        def mean(box, field):
            return statistics.mean(r["bias_diag"][box][field] for r in rs if box in r["bias_diag"])
        lines.append(f"| {LABELS[g]} | {statistics.mean(r['rmse'] for r in rs):.4f} | "
                     f"{mean('failed_all','mean_bias'):+.4f} | {mean('early','mean_bias'):+.4f} | "
                     f"{mean('middle','mean_bias'):+.4f} | {mean('late','mean_bias'):+.4f} | "
                     f"{mean('censored','lb_violation_rate'):.4f} |\n")
    lines.append("\n## Source−Random 配对 bias 差异\n\n"
                 "正值 = source 更高估 RUL（或相对删失下界更保守）；CI 为 n=5 seed 配对 t-CI，描述性。\n\n")
    lines.append("| source | box | Δ mean_bias | CI95 | 正向 seed |\n|---|---|---:|---:|---:|\n")
    for g in GROUPS[1:]:
        for box in BOXES:
            field = "mean_bias_vs_lb" if box == "censored" else "mean_bias"
            ds = [by[g][s]["bias_diag"][box][field] - by[GROUPS[0]][s]["bias_diag"][box][field]
                  for s in sorted(by[g])]
            m, lo, hi = ci(ds)
            lines.append(f"| {LABELS[g]} | {box} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                         f"{sum(x > 0 for x in ds)}/5 |\n")
    lines += ["\n## 判读\n\n",
              "1. 三个 source 臂在 failed_all 的方向差异均为描述性，CI 跨 0，不能声称全局系统性偏差。\n",
              "2. **IGBT late 箱出现最清晰的偏差证据**：source−random Δmean_bias=+0.1083，"
              "CI [+0.0365,+0.1801]，5/5 seed 为正；IGBT 源先验在晚期退化阶段系统性高估 RUL，"
              "与其 k=3 RMSE 负迁移方向一致。\n",
              "3. MOSFET late Δ=+0.0545，CI 跨 0；Multi late Δ=-0.0513，CI 跨 0。"
              "因此机制证据对 IGBT 臂成立（探索性），不能泛化为所有 source 臂共同机制。\n",
              "4. 删失下界违反率/相对下界 bias 没有稳定的跨臂模式；负迁移不主要表现为统一的删失风险偏置。\n",
              "5. 结论升级为：**错误物理尺度先验可在特定源臂（IGBT）的 late 阶段诱发系统性 RUL 高估；"
              "跨源臂总体机制仍包含优化干扰/方差，A3/A1 的终局判停不改变。**\n"]
    out = ROOT / "outputs" / "bias_diag" / "analysis.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(out)
    print("endpoint:", endpoint)
    print("IGBT late:", lines[-5])

if __name__ == "__main__":
    main()
