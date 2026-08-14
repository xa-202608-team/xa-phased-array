#!/usr/bin/env python
"""scripts/run_arch_aligned.py — 架构对齐迁移主矩阵

消除原始主矩阵中的架构混淆变量 (GRU target_only 0.2297 vs TCN 迁移臂 0.2503)。
补齐 {GRU, TCN} × {source_frozen, random_frozen} 2×2 矩阵, 在同架构内报配对差值。

已有 (主矩阵, TCN): ch_source_pretrain_frozen 0.2503 / ch_random_frozen 0.2510
新增 (本脚本): ch_source_frozen_gru / ch_random_frozen_gru

配对分析:
  - 源权重贡献 (架构对齐后): source_frozen − random_frozen, 分 GRU/TCN
  - 架构贡献: GRU − TCN (source frozen 下 + target only 下)

用法:
  python scripts/run_arch_aligned.py [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                                          # noqa: E402
from src.experiments.run_groups import run_one_group, _paired_delta_ci     # noqa: E402

CKPT_DIR = ROOT / "checkpoints"


def main():
    ap = argparse.ArgumentParser(description="架构对齐迁移主矩阵 2×2")
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)

    # 2×2 矩阵: {gru, tcn} × {source_frozen, random_frozen}
    arms = [
        ("ch_source_frozen_gru", "source_finetune", "gru"),
        ("ch_random_frozen_gru", "random_frozen", "gru"),
        ("ch_source_frozen_tcn", "source_finetune", "tcn"),
        ("ch_random_frozen_tcn", "random_frozen", "tcn"),
    ]

    by = {}
    for tag, mode, enc in arms:
        by[tag] = []
        for s in range(args.seeds):
            seed = cfg["seed"] + s
            try:
                m = run_one_group(
                    mode, seed, cfg, smoke=False,
                    encoder_override=enc,
                    component="phased_array",
                    group_name=tag, level="channel", k_shot="all")
                by[tag].append(m)
                print(f">> {tag:30s} seed{seed} ({enc}): "
                      f"RMSE={m['rmse']:.4f} PHM={m['phm']:.2f} MAE={m['mae']:.4f}")
            except Exception as exc:                                    # noqa: BLE001
                print(f"!! {tag} seed{seed} 失败: {exc}")

    # 聚合
    agg = {}
    for tag, ms in by.items():
        if not ms:
            continue
        rmses = [m["rmse"] for m in ms]
        agg[tag] = {
            "rmse_mean": statistics.mean(rmses),
            "rmse_std": statistics.stdev(rmses) if len(rmses) > 1 else 0.0,
            "phm_mean": statistics.mean(m["phm"] for m in ms),
            "mae_mean": statistics.mean(m["mae"] for m in ms),
            "n": len(ms),
            "encoder": ms[0]["encoder"],
            "by_seed": {str(m["seed"]): m["rmse"] for m in ms},
        }

    # 配对分析 (架构对齐后)
    paired = {}
    for enc in ["gru", "tcn"]:
        src_tag = f"ch_source_frozen_{enc}"
        rnd_tag = f"ch_random_frozen_{enc}"
        if src_tag in agg and rnd_tag in agg:
            # _paired_delta_ci(tgt, src) = tgt - src; 传 (source, random) = source - random
            # 负 → source RMSE 更低 (源初始化有益)
            ps = _paired_delta_ci(
                {int(k): v for k, v in agg[src_tag]["by_seed"].items()},
                {int(k): v for k, v in agg[rnd_tag]["by_seed"].items()})
            paired[f"source_vs_random_{enc}"] = ps

    # 跨架构: GRU source vs TCN source
    if "ch_source_frozen_gru" in agg and "ch_source_frozen_tcn" in agg:
        ps = _paired_delta_ci(
            {int(k): v for k, v in agg["ch_source_frozen_gru"]["by_seed"].items()},
            {int(k): v for k, v in agg["ch_source_frozen_tcn"]["by_seed"].items()})
        paired["source_gru_vs_tcn"] = ps

    result = {"arms": agg, "paired": paired, "n_seeds": args.seeds}
    out_path = CKPT_DIR / "arch_aligned_results.json"
    CKPT_DIR.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    # 打印摘要
    print("\n" + "=" * 70)
    print("架构对齐 2×2 矩阵 (channel level, 200 traj, k=all)")
    print(f"{'臂':30s} {'enc':6s} {'RMSE':16s} {'PHM':8s} {'n':3s}")
    print("-" * 70)
    for tag in sorted(agg.keys()):
        a = agg[tag]
        print(f"{tag:30s} {a['encoder']:6s} "
              f"{a['rmse_mean']:.4f}±{a['rmse_std']:.4f}  "
              f"{a['phm_mean']:>7.0f}  {a['n']:3d}")

    print("\n配对分析 (Δ = target − source, 负值 = target RMSE 更低 = 正向):")
    for key, ps in paired.items():
        if ps:
            cross = "跨0" if ps["ci_crosses_zero"] else "不跨0"
            print(f"  {key:30s}: Δ={ps['delta_mean']:+.4f} ± {ps['delta_std']:.4f}, "
                  f"CI [{ps['ci95_lo']:+.4f}, {ps['ci95_hi']:+.4f}], {cross}")

    print(f"\n结果保存到: {out_path}")


if __name__ == "__main__":
    main()
