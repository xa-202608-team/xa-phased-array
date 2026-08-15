"""Phase 2 集成验证: A1+A2 (hinge + isotonic) + D (memory bank) 综合后的 5 种子基线。

每种子 reset_global_memory_bank() 防 D 的全局 bank 跨种子累积污染。
对比之前完整归一+ES 基线 (target 0.161 / src_pretrain 0.158 / src_mmd 0.153),
验证正确性修复后数字 (预期: 源 encoder 质量改善但迁移增益未明显提升, 因 D2 adapter
瓶颈未解)。

用法: python scripts/phase2_verify.py (后台 ~30min)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.utils import load_config
from src.experiments.run_groups import run_one_group
from src.transfer.mmd import reset_global_memory_bank

cfg = load_config(ROOT / "configs/phased_array.yaml")
groups = [("target_only", "target_only_tcn"),
          ("source_finetune", "source_pretrain_finetune"),
          ("source_mmd_finetune", "source_mmd_physics")]
seeds = [42, 43, 44, 45, 46]
results = {}

for mode, gname in groups:
    rmses = []
    for seed in seeds:
        reset_global_memory_bank()          # 每种子清 bank (D 的全局 bank 防跨种子污染)
        m = run_one_group(mode, seed, cfg, smoke=False,
                          component="phased_array", group_name=gname)
        rmses.append(m["rmse"])
        print(f"  {gname} seed{seed}: {m['rmse']:.4f}", flush=True)
    mean, std = float(np.mean(rmses)), float(np.std(rmses))
    results[gname] = (mean, std)
    print(f">>> {gname}: mean={mean:.4f} std={std:.4f}", flush=True)

print("\n=== 迁移增益 (target - source) ===", flush=True)
t = results["target_only_tcn"][0]
sp = results["source_pretrain_finetune"][0]
sm = results["source_mmd_physics"][0]
print(f"target={t:.4f}  src_pretrain={sp:.4f} (gain {t - sp:+.4f})  "
      f"src_mmd={sm:.4f} (gain {t - sm:+.4f})", flush=True)
