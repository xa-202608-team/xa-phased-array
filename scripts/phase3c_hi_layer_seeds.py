"""Phase 3c: HI 层迁移增益 5 种子验证 (target_only vs source_mmd, HI 动力学层)。

P0-1 修复后: 直接 import + 调用 run_hi_layer 统一入口 (不再 subprocess),
确保训练逻辑与 train_transfer.py 完全一致 (val early-stop + target/source 同预算)。
run_hi_layer 内部每 seed reset MMD bank + set_seed, 天然隔离。
对比 Phase 2 旧观测层 (target 0.161 / source_mmd 0.156, +0.005 不显著)。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.transfer.train_transfer import run_hi_layer   # noqa: E402

METRICS = ROOT / "checkpoints" / "transfer_metrics_phased_array_hilayer.json"
SEEDS = [42, 43, 44, 45, 46]


def _make_args(seed, target_only, config="configs/phased_array.yaml",
               smoke=False, ckpt=None):
    """构造 run_hi_layer 所需的 argparse namespace (与 CLI --hi-layer 等价)。"""
    return argparse.Namespace(
        config=config, ckpt=ckpt, smoke=smoke,
        hi_layer=True, target_only=target_only, seed=seed)


def main(smoke=False, seeds=None):
    _seeds = seeds or SEEDS
    results = {}
    for mode, target_only in [("target_only", True), ("source_mmd", False)]:
        rmses = []
        for seed in _seeds:
            args = _make_args(seed, target_only, smoke=smoke)
            run_hi_layer(args)             # 统一入口: val early-stop + 同预算
            m = json.loads(METRICS.read_text(encoding="utf-8"))
            r = m.get("test_rul_rmse", m.get("val_rul_rmse"))
            rmses.append(r)
            print(f"  {mode} seed{seed}: test={r:.4f}", flush=True)
        mean, std = float(np.mean(rmses)), float(np.std(rmses, ddof=1))  # 样本 std (n-1), 与 run_groups.statistics.stdev 一致
        results[mode] = (mean, std, rmses)
        print(f">>> HI层 {mode}: mean={mean:.4f} std={std:.4f}", flush=True)

    print("\n=== HI 层迁移增益 (target_only - source_mmd) ===", flush=True)
    t = results["target_only"][0]
    s = results["source_mmd"][0]
    print(f"target_only={t:.4f}  source_mmd={s:.4f}  gain={t - s:+.4f}", flush=True)
    print(f"\n(对比旧观测层 Phase2: target 0.161 / source_mmd 0.156 / gain +0.005)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HI 层 5 种子验证 (调用 run_hi_layer 统一入口)")
    ap.add_argument("--smoke", action="store_true", help="快速验证 (2 epoch)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="覆盖默认 5 种子")
    args = ap.parse_args()
    main(smoke=args.smoke, seeds=args.seeds)
