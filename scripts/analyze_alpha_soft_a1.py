"""A1 α-soft 剂量-响应分析 (§4d 预注册判读的固化脚本).

输入: outputs/alpha_soft_a1/results_partial.jsonl (30 组-seed) +
      outputs/kshot_source_arms/all_metrics_phased_array.json (§4c 端点核对).
输出: 剂量-响应表 (val/test) + α* (argmin val mean) + 各 α vs α=0 配对 CI +
      端点 bit-exact 校验 + 预注册判读结论.

统计纪律: 与 §4c/§3.4 同款 — per-seed paired Δ, 样本 std + t(n-1) 临界值,
n<10 标 exploratory_only.
"""
import json
import math
import statistics
import sys
from pathlib import Path

from scipy import stats as sp

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "outputs/alpha_soft_a1/results_partial.jsonl"
KSHOT_REF = ROOT / "outputs/kshot_source_arms/all_metrics_phased_array.json"

GROUPS = {  # 组名 → α 剂量
    "ch_random_full_finetune_k3": 0.0,
    "ch_alpha_soft_a005_k3": 0.05,
    "ch_alpha_soft_a010_k3": 0.10,
    "ch_alpha_soft_a025_k3": 0.25,
    "ch_alpha_soft_a050_k3": 0.50,
    "ch_source_mmd_physics_k3": 1.0,
}


def paired_ci(deltas):
    n = len(deltas)
    md = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    crit = float(sp.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(max(n, 1))
    return md, sd, md - crit * se, md + crit * se


def main():
    recs = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 端点 bit-exact 校验 (vs §4c per-seed raw)
    if KSHOT_REF.exists():
        old_raw = json.loads(KSHOT_REF.read_text(encoding="utf-8"))["agg"]["_raw_by"]
        for gname, label in [("ch_random_full_finetune_k3", "α=0 (random)"),
                             ("ch_source_mmd_physics_k3", "α=1 (MOSFET)")]:
            new = {r["seed"]: r["rmse"] for r in recs if r["group"] == gname}
            old = old_raw.get(gname, {})
            ok = bool(old) and all(new.get(int(s)) == v for s, v in old.items())
            print(f"[端点校验] {label}: bit-exact={ok} ({len(new)} vs {len(old)} seeds)")

    data = {a: {} for a in GROUPS.values()}
    for r in recs:
        if r["group"] in GROUPS:
            data[GROUPS[r["group"]]][r["seed"]] = (r["val_rmse"], r["rmse"])

    print("\n[剂量-响应] alpha | val_rmse | test_rmse | per-seed test")
    for a in sorted(data):
        vs = [v for v, _ in data[a].values()]
        ts = [t for _, t in data[a].values()]
        print(f"  {a:>4.2f} | {statistics.mean(vs):.4f} | {statistics.mean(ts):.4f} | "
              + " ".join(f"{t:.4f}" for t in ts))
    alpha_star = min(data, key=lambda a: statistics.mean(v for v, _ in data[a].values()))
    print(f"  α* (argmin 5-seed val mean) = {alpha_star}")

    print("\n[配对 ΔRMSE] (α − α=0, 正 = 更差)")
    base = {s: t for s, (_, t) in data[0.0].items()}
    sig_any = False
    for a in sorted(data):
        if a == 0.0:
            continue
        deltas = [data[a][s][1] - base[s] for s in sorted(base) if s in data[a]]
        md, sd, lo, hi = paired_ci(deltas)
        crosses = lo <= 0 <= hi
        verdict = ("跨0" if crosses else
                   ("全正:显著更差" if md > 0 else "全负:显著更优"))
        if not crosses and md < 0:
            sig_any = True
        print(f"  α={a:.2f}: Δ={md:+.4f} ±{sd:.4f}, CI95 [{lo:+.4f},{hi:+.4f}] → {verdict}"
              + ("  [exploratory, n<10]" if len(deltas) < 10 else ""))

    md, sd, lo, hi = paired_ci([data[alpha_star][s][1] - base[s] for s in sorted(base)])
    print(f"\n[判读输入] α*={alpha_star} test vs α=0: Δ={md:+.4f}, CI95 [{lo:+.4f},{hi:+.4f}]")
    print("[判读] " + ("剂量控制翻正 (CI 全负, 预注册条款1触发)" if sig_any else
                       "未翻正: 无中间 α 显著优于 random → 判停条款触发, A1 并入 A2/A3 汇总判停"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
