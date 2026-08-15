#!/usr/bin/env python
"""scripts/run_kshot_frozen.py — K-shot 少样本适配曲线（frozen 臂）

评审建议: 原图 pa_05 用 ch_source_mmd_physics（S3 全微调, 劣势协议）代表迁移,
导致全 k 负迁移。应改用 ch_source_pretrain_frozen（冻结 encoder, 小样本最占优区间）。

本脚本跑 ch_target_only_gru + ch_source_pretrain_frozen 在 k=1,3,5,all 下的 RMSE,
供更新 K-shot 曲线图。

用法:
  python scripts/run_kshot_frozen.py [--seeds 3]
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
from src.experiments.run_groups import run_one_group                       # noqa: E402

CKPT_DIR = ROOT / "checkpoints"


def main():
    ap = argparse.ArgumentParser(description="K-shot frozen 臂实验")
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)

    k_shots = [1, 3, 5, "all"]
    arms = [
        ("ch_target_only_gru", "target_only", "gru"),
        ("ch_source_pretrain_frozen", "source_finetune", "tcn"),
    ]

    results = {}
    for tag, mode, enc in arms:
        for k in k_shots:
            key = f"{tag}_k{k}"
            results[key] = []
            for s in range(args.seeds):
                seed = cfg["seed"] + s
                try:
                    m = run_one_group(
                        mode, seed, cfg, smoke=False,
                        encoder_override=enc,
                        component="phased_array",
                        group_name=key, level="channel", k_shot=k)
                    results[key].append(m["rmse"])
                    print(f">> {key:35s} seed{seed}: RMSE={m['rmse']:.4f}")
                except Exception as exc:                                    # noqa: BLE001
                    print(f"!! {key} seed{seed} 失败: {exc}")

    # 保存
    out = CKPT_DIR / "kshot_frozen_results.json"
    CKPT_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")

    # 摘要表
    print("\n" + "=" * 60)
    print("K-shot (frozen 臂) 摘要:")
    print(f"{'k':6s} {'target_gru':16s} {'source_frozen':16s} {'gain':10s}")
    print("-" * 60)
    summary = {}
    for k in k_shots:
        t_key = f"ch_target_only_gru_k{k}"
        s_key = f"ch_source_pretrain_frozen_k{k}"
        t_vals = results.get(t_key, [])
        s_vals = results.get(s_key, [])
        if t_vals and s_vals:
            t_mean = statistics.mean(t_vals)
            s_mean = statistics.mean(s_vals)
            gain = t_mean - s_mean
            summary[f"k{k}"] = {"target": t_mean, "source_frozen": s_mean, "gain": gain}
            print(f"{str(k):6s} {t_mean:.4f}           {s_mean:.4f}            {gain:+.4f}")
        else:
            print(f"{str(k):6s} {'—':16s} {'—':16s}")

    out_summary = CKPT_DIR / "kshot_frozen_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                           encoding="utf-8")
    print(f"\n结果保存到: {out}")
    print(f"摘要保存到: {out_summary}")


if __name__ == "__main__":
    main()
