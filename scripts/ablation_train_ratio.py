"""少样本消融 v2: 3 组 (target_only / source_pretrain / source_mmd) × 3 档 train ratio。

v1 只跑 target_only + source_mmd, 发现迁移增益未随少样本放大 (half 反超, few S3 发散)。
v2 补 source_pretrain_finetune (S2 only, 无 MMD 发散) — 验证 S2-only 少样本是否更稳;
加 median 统计防离群发散; 修 Unicode (GBK 控制台)。

验证迁移学习核心卖点: 数据稀缺时源域预训练的真正价值。
(区别于 plan §8 telemetry_sparsity 遥测稀疏; 本脚本消融 train 轨迹数)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.utils import load_config
from src.experiments.run_groups import run_one_group

cfg = load_config(ROOT / "configs/phased_array.yaml")
ratios = [(0.15, "full", 30), (0.05, "half", 10), (0.015, "few", 3)]
seeds = [42, 43, 44]
groups = [("target_only", "target_only_tcn"),
          ("source_finetune", "source_pretrain_finetune"),
          ("source_mmd_finetune", "source_mmd_physics")]
results = {}

for tr_ratio, label, _ in ratios:
    cfg["transfer"]["split"]["train"] = tr_ratio
    cfg["transfer"]["split"]["val"] = 0.20
    cfg["transfer"]["split"]["test"] = round(1.0 - tr_ratio - 0.20, 3)
    print(f"\n===== train={label} (ratio={tr_ratio}) =====", flush=True)
    for mode, gname in groups:
        rmses = []
        for seed in seeds:
            try:
                m = run_one_group(mode, seed, cfg, smoke=False,
                                  component="phased_array", group_name=gname)
                rmses.append(m["rmse"])
                flag = " (发散>1)" if m["rmse"] > 1.0 else ""
                print(f"  {label} {gname} seed{seed}: {m['rmse']:.4f}{flag}", flush=True)
            except Exception as e:    # noqa: BLE001
                print(f"  {label} {gname} seed{seed} 失败: {e}", flush=True)
        results[(label, gname)] = rmses
        med = float(np.median(rmses)) if rmses else float("nan")
        mean = float(np.mean(rmses)) if rmses else float("nan")
        diverged = sum(1 for r in rmses if r > 1.0)
        print(f">>> {label} {gname}: mean={mean:.4f} median={med:.4f}"
              f"{f'  [{diverged} 发散]' if diverged else ''}", flush=True)

print("\n===== 迁移增益 (median: target - source) =====", flush=True)
print(f"{'train':<8}{'target':>10}{'src_pretrain':>15}{'src_mmd':>11}"
      f"{'gain(S2)':>11}{'gain(MMD)':>11}", flush=True)
for label, _, _ in ratios:
    t = float(np.median(results.get((label, "target_only_tcn"), [np.nan])))
    sp = float(np.median(results.get((label, "source_pretrain_finetune"), [np.nan])))
    sm = float(np.median(results.get((label, "source_mmd_physics"), [np.nan])))
    print(f"{label:<8}{t:>10.4f}{sp:>15.4f}{sm:>11.4f}{t - sp:>+11.4f}{t - sm:>+11.4f}",
          flush=True)
