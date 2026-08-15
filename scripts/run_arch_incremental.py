#!/usr/bin/env python
"""scripts/run_arch_incremental.py — 架构对齐实验（增量保存版）

每个实验完成后立即追加到 JSONL 文件, 防超时中断丢数据。
跑完后读取全部结果做聚合 + 配对 CI。

用法:
  python scripts/run_arch_incremental.py --seeds 43,44
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                                      # noqa: E402
from src.experiments.run_groups import run_one_group, _paired_delta_ci  # noqa: E402

CKPT_DIR = ROOT / "checkpoints"
OUT_JSONL = CKPT_DIR / "arch_aligned_raw.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--seeds", default="43,44",
                    help="逗号分隔 seed 列表 (42 已跑完, 默认补 43,44)")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    CKPT_DIR.mkdir(exist_ok=True)
    arms = [
        ("ch_source_frozen_gru", "source_finetune", "gru"),
        ("ch_random_frozen_gru", "random_frozen", "gru"),
        ("ch_source_frozen_tcn", "source_finetune", "tcn"),
        ("ch_random_frozen_tcn", "random_frozen", "tcn"),
    ]

    for seed in seeds:
        for tag, mode, enc in arms:
            # 跳过已跑过的
            existing = set()
            if OUT_JSONL.exists():
                for line in OUT_JSONL.read_text(encoding="utf-8").strip().split("\n"):
                    if line:
                        try:
                            d = json.loads(line)
                            existing.add((d["tag"], d["seed"]))
                        except json.JSONDecodeError:
                            pass
            if (tag, seed) in existing:
                print(f"[skip] {tag} seed{seed} 已存在", flush=True)
                continue
            t0 = time.time()
            try:
                m = run_one_group(
                    mode, seed, cfg, smoke=False,
                    encoder_override=enc, component="phased_array",
                    group_name=f"{tag}_seed{seed}", level="channel", k_shot="all")
                result = {
                    "tag": tag, "seed": seed, "encoder": enc,
                    "rmse": m["rmse"], "phm": m["phm"], "mae": m["mae"],
                    "mode": mode,
                }
                with open(OUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"DONE: {tag} seed{seed} ({enc}) RMSE={m['rmse']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as exc:                                    # noqa: BLE001
                print(f"FAIL: {tag} seed{seed}: {exc}", flush=True)

    # 聚合全部结果
    aggregate_all()


def aggregate_all():
    """读取 JSONL 全部结果, 聚合 + 配对 CI, 写 summary。"""
    if not OUT_JSONL.exists():
        print("无结果文件")
        return
    by = {}
    for line in OUT_JSONL.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        by.setdefault(d["tag"], []).append(d)

    agg = {}
    for tag, records in by.items():
        rmses = [r["rmse"] for r in records]
        if not rmses:
            continue
        agg[tag] = {
            "rmse_mean": statistics.mean(rmses),
            "rmse_std": statistics.stdev(rmses) if len(rmses) > 1 else 0.0,
            "n": len(rmses),
            "by_seed": {r["seed"]: r["rmse"] for r in records},
        }

    # 配对分析
    paired = {}
    for enc in ["gru", "tcn"]:
        src = f"ch_source_frozen_{enc}"
        rnd = f"ch_random_frozen_{enc}"
        if src in agg and rnd in agg:
            ps = _paired_delta_ci(agg[src]["by_seed"], agg[rnd]["by_seed"])
            paired[f"source_vs_random_{enc}"] = ps
    # 跨架构
    for proto in ["source_frozen", "random_frozen"]:
        g = f"ch_{proto}_gru"
        t = f"ch_{proto}_tcn"
        if g in agg and t in agg:
            ps = _paired_delta_ci(agg[g]["by_seed"], agg[t]["by_seed"])
            paired[f"{proto}_gru_vs_tcn"] = ps

    summary = {"arms": agg, "paired": paired}
    out = CKPT_DIR / "arch_aligned_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")

    print("\n" + "=" * 70)
    print("架构对齐 2×2 矩阵汇总:")
    print(f"{'臂':30s} {'RMSE':16s} {'n':3s}")
    print("-" * 70)
    for tag in sorted(agg.keys()):
        a = agg[tag]
        print(f"{tag:30s} {a['rmse_mean']:.4f}±{a['rmse_std']:.4f}  {a['n']:3d}")
    print("\n配对 (Δ = target − source, 负 = target 更低 = 正向):")
    for key, ps in paired.items():
        if ps:
            cross = "跨0" if ps["ci_crosses_zero"] else "不跨0"
            print(f"  {key:30s}: Δ={ps['delta_mean']:+.4f} ± {ps['delta_std']:.4f}, "
                  f"CI [{ps['ci95_lo']:+.4f}, {ps['ci95_hi']:+.4f}], {cross}")
    print(f"\n汇总保存到: {out}")


if __name__ == "__main__":
    main()
