"""B 路线 sim_v1→sim_v2 剂量分析 (§4e 预注册判读的固化脚本).

输入: outputs/simv1_transfer_b/results_partial.jsonl (20 组-seed) +
      outputs/alpha_soft_a1/results_partial.jsonl (跨矩阵参照: 同协议 MOSFET 异构源) +
      outputs/kshot_source_arms/all_metrics_phased_array.json (§4c 端点参照).
输出: 两口径 (k=all / k=3) 的 simv1−random 配对 CI + §4e 三分支判读 +
      跨矩阵方向对比 (异构源 vs 同构源).

统计纪律: per-seed paired Δ, 样本 std + t(n-1), n<10 标 exploratory.
"""
import json
import math
import statistics
import sys
from pathlib import Path

from scipy import stats as sp

ROOT = Path(__file__).resolve().parents[1]
B_JSONL = ROOT / "outputs/simv1_transfer_b/results_partial.jsonl"
A1_JSONL = ROOT / "outputs/alpha_soft_a1/results_partial.jsonl"


def load(jsonl):
    data = {}
    for l in jsonl.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            data.setdefault(r["group"], {})[r["seed"]] = (r["val_rmse"], r["rmse"])
    return data


def paired_ci(data, ga, gb):
    """ga − gb per-seed 配对 (正 = ga 更差)。"""
    ds = [data[ga][s][1] - data[gb][s][1] for s in sorted(data[ga]) if s in data[gb]]
    n = len(ds)
    md = statistics.mean(ds)
    sd = statistics.stdev(ds) if n > 1 else 0.0
    crit = float(sp.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(max(n, 1))
    return md, sd, md - crit * se, md + crit * se, sum(1 for d in ds if d < 0)


def main():
    data = load(B_JSONL)
    print("[B 矩阵] simv1 vs random (v1-MMD 窗同协议, 唯一差异=初始化)")
    results = {}
    for k in ["kall", "k3"]:
        gs, gr = f"ch_simv1_source_{k}", f"ch_random_full_finetune_{k}"
        vs = statistics.mean(v for v, _ in data[gs].values())
        vr = statistics.mean(v for v, _ in data[gr].values())
        ts = statistics.mean(t for _, t in data[gs].values())
        tr = statistics.mean(t for _, t in data[gr].values())
        md, sd, lo, hi, npos = paired_ci(data, gs, gr)
        crosses = lo <= 0 <= hi
        verdict = ("跨0" if crosses else ("全负: simv1 显著更优" if md < 0 else "全正: simv1 显著更差"))
        results[k] = (md, lo, hi, crosses)
        print(f"  [{k:>4}] simv1 val={vs:.4f} test={ts:.4f} | random val={vr:.4f} test={tr:.4f}")
        print(f"         Δ(simv1−random)={md:+.4f} ±{sd:.4f}, CI95 [{lo:+.4f},{hi:+.4f}], "
              f"正向 seed {npos}/5 → {verdict}  [exploratory, n<10]")

    # §4e 三分支判读
    kall_neg = not results["kall"][3] and results["kall"][0] < 0
    if kall_neg:
        verdict = "分支1: B 翻正 (k=all CI 全负)"
    elif not results["kall"][3] and not results["k3"][3]:
        verdict = "分支3: 双口径均显著 (非跨0) — 如实报告方向"
    else:
        verdict = "分支3: 双口径 CI 跨 0 → sim-to-real 代理封口 (判停触发)"
    print(f"\n[判读] {verdict}")

    # 跨矩阵方向对比: 同协议同 seed 下, 异构源 (MOSFET, A1 矩阵端点) vs 同构源 (simv1)
    if A1_JSONL.exists():
        a1 = load(A1_JSONL)
        md_m, sd_m, lo_m, hi_m, np_m = paired_ci(
            a1, "ch_source_mmd_physics_k3", "ch_random_full_finetune_k3")
        b3 = results["k3"]
        print(f"\n[跨矩阵对比 k=3] MOSFET 异构源 Δ={md_m:+.4f} vs simv1 同构源 Δ={b3[0]:+.4f}")
        print("  (两矩阵 random 臂 MMD 窗不同: MOSFET vs v1 — 方向对比仅作机制参考)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
