# -*- coding: utf-8 -*-
"""完整端到端复现（reproduce_full，契约 §2 v1.1）。

保留 200 轨迹 + 冻结 5 seeds（42..46）语义；**任何主步骤失败必须非零退出**，
并移除旧 entrypoint 的 `|| echo` 吞错模式（P5 迁移训练 / P6 基线 / P7 合并）。

流程：
  1. prepare_source_data（canonical NASA / synthetic，manifest 标 source_mode）
  2. 源域预训练（canonical 存在 -> --canonical；synthetic -> 默认 schema_v3 合成路径）
  3. generate_simulation（200 轨迹，sim_v2 + sim_v1，seed=42，写实际 config SHA256）
  4. build_channel_hi / build_array_hi
  5. run_groups --level channel（ch_* + cross_level 层级消融，5 seeds）
     + --export-inference-dir（F1 批3: ch_target_only_gru seed42 val-best bundle）
  5.5 predict_gru 推理自检（bundle 整链: 无标签读→factory 重建→rul_prediction.json
     经 rul-prediction.schema.json 校验; 失败即失败）
  6. channel_baselines（非学习基线，失败即失败）
  7. Schema 校验 + manifest.json / metrics.json / run.log / REPRODUCE_OK

用法：
  python scripts/reproduce_full.py --output outputs/full
  bash scripts/reproduce_full.sh --output outputs/full
  pwsh scripts/reproduce_full.ps1 -Output outputs/full
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reproduce_judge import RunLog, _git_commit, _sha256, _validate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/full", help="输出目录")
    ap.add_argument("--config", default=str(ROOT / "configs/phased_array.yaml"))
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=None,
                    help="对比实验种子数（默认 5；--fast 调试模式默认 1——4 轨迹小样本"
                         "下 0.15 train 比例遇个别种子会抽空）")
    ap.add_argument("--fast", action="store_true",
                    help="调试用小规模（非完整复现，输出目录建议区分）")
    args = ap.parse_args()
    if args.seeds is None:
        args.seeds = 1 if args.fast else 5

    out_dir = (ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir)

    import time
    t_start = time.time()
    commit = _git_commit()
    config_path = Path(args.config).resolve()
    py = sys.executable
    ckpt = ROOT / "checkpoints/source_phased_array_tcn_pretrain.pt"

    log.log(f"reproduce_full start; git_commit={commit}; fast={args.fast}")

    # 1. 源域准备
    log.step("P1 源域数据准备")
    source_report_path = out_dir / "source_report.json"
    log.run([py, "scripts/prepare_source_data.py", "--config", str(config_path),
             "--output", str(source_report_path)])
    source_report = json.loads(source_report_path.read_text("utf-8"))
    source_mode = source_report["source_mode"]
    log.log(f"source_mode={source_mode}")
    (out_dir / "source_mode.txt").write_text(
        f"source_mode={source_mode}\n"
        f"note={'synthetic 源域：结果不得与正式源域结果混用' if source_mode == 'synthetic' else 'canonical/真实源域'}\n",
        encoding="utf-8")

    # 2. 源域预训练（ckpt 已存在则跳过；P1 已保证 canonical H5 存在，
    #    三种 source_mode 统一走 --canonical；synthetic 结果由 manifest 如实标注）
    log.step("P2 源域预训练 (TCN 双头)")
    if ckpt.is_file():
        log.log(f"  >> 跳过：{ckpt.relative_to(ROOT)} 已存在")
    else:
        log.run([py, "-m", "src.train.pretrain", "--config", str(config_path),
                 "--canonical"])

    # 3. 仿真（200 轨迹双仿真集）
    log.step(f"P3 三级链仿真 ({'fast' if args.fast else args.n_traj} 轨迹, seed=42)")
    sim_manifest_path = out_dir / "sim_manifest.json"
    sim_cmd = [py, "scripts/generate_simulation.py", "--config", str(config_path),
               "--seed", "42", "--manifest", str(sim_manifest_path)]
    if args.fast:
        sim_cmd.append("--fast")
    else:
        sim_cmd += ["--n_traj", str(args.n_traj)]
    log.run(sim_cmd)
    sim_manifest = json.loads(sim_manifest_path.read_text("utf-8"))

    # 4. HI 构造
    log.step("P4 通道级 + 服务级 HI 构造")
    log.run([py, "-m", "src.sim.build_channel_hi", "--config", str(config_path), "--report"])
    log.run([py, "-m", "src.sim.build_array_hi", "--config", str(config_path), "--report"])

    # 5. 对比实验（正式 5 seeds；run_groups 内部已失败传播 -> 非零退出；
    #    Windows 本机退出期崩溃 0xC0000409 以产物哨兵兜底，见 RunLog）
    #    F1 批3: 同步导出推理 bundle (ch_target_only_gru seed42 val-best) 供 5.5 自检
    log.step(f"P5 对比实验 (level=channel, {args.seeds} seed{'s' if args.seeds > 1 else ''})")
    groups_dir = out_dir / "groups"
    metrics_json = groups_dir / "all_metrics_phased_array.json"
    bundle_dir = groups_dir / "inference_bundle"
    log.run([py, "-m", "src.experiments.run_groups", "--config", str(config_path),
             "--seeds", str(args.seeds), "--level", "channel",
             "--export-inference-dir", str(bundle_dir),
             "--output-dir", str(groups_dir)], sentinel=metrics_json)
    all_metrics = json.loads(metrics_json.read_text("utf-8"))

    # 5.5 推理 bundle 自检 (F1 批3 依赖闭合: 导出→重建→预测→schema 校验, 失败即失败)
    import yaml
    cfg_doc = yaml.safe_load(config_path.read_text("utf-8"))
    ch_h5 = ROOT / cfg_doc["channel_level"]["feature_path"]
    log.step("P5.5 推理 bundle 自检 (predict_gru, limit 4 通道)")
    pred_dir = out_dir / "inference"
    log.run([py, "-m", "component.predict_gru",
             "--features", str(ch_h5), "--bundle-dir", str(bundle_dir),
             "--output", str(pred_dir), "--stride", "50", "--limit-channels", "4"])
    rul_pred = json.loads((pred_dir / "rul_prediction.json").read_text("utf-8"))
    log.log(f"rul_prediction: {len(rul_pred['predictions'])} predictions, "
            f"model={rul_pred['model']['group']} seed{rul_pred['model']['seed']}")

    # 6. 通道级基线（旧 entrypoint `|| echo` 吞错在此修复：失败即失败）
    log.step("P6 通道级非学习基线")
    baselines_path = out_dir / "baselines_channel.json"
    log.run([py, "-m", "src.baselines.channel_baselines", "--config", str(config_path),
             "--out", str(baselines_path)])
    baselines = json.loads(baselines_path.read_text("utf-8"))

    # 7. 汇总
    log.step("P7 指标汇总与 Schema 校验")
    metrics: dict[str, float] = {}
    for group, agg in all_metrics.get("agg", {}).items():
        if isinstance(agg, dict) and "rmse_mean" in agg:
            metrics[f"rmse_mean.{group}"] = round(float(agg["rmse_mean"]), 6)
    for name, payload in baselines.items():
        if isinstance(payload, dict) and "rmse" in payload:
            metrics[f"baseline.{name}.rmse"] = round(float(payload["rmse"]), 6)
    # F1 批3: 推理链自检产物计数 (>0 证明 bundle→predict_gru→schema 全链通)
    metrics["inference.rul_predictions"] = float(len(rul_pred["predictions"]))
    if not metrics:
        log.log("!! 未提取到任何指标")
        raise SystemExit(1)
    metrics_doc = {
        "schema_version": "1.0.0",
        "component": "phased_array",
        "run_id": f"full-{source_mode}-seeds42-{41 + args.seeds}",
        "mode": "full",
        "metrics": metrics,
        "conclusion": "NO_POSITIVE_TRANSFER_SUPPORTED",
    }
    _validate(metrics_doc, "metrics.schema.json", "metrics.json")
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = round(time.time() - t_start, 1)
    manifest = {
        "schema_version": "1.0.0",
        "component": "phased_array",
        "version": "phased-array-v0.1.0",
        "git_commit": commit,
        "contract_version": "component-contract-v1.1.0",
        "config_sha256": sim_manifest["config_sha256"],
        "data": [
            {"name": run["dynamics_id"], "version": run["tag"],
             "sha256": run["h5_sha256"], "redistributable": True}
            for run in sim_manifest["runs"]
        ] + [
            {"name": "mosfet-canonical-source", "version": source_mode,
             "sha256": source_report["canonical_sha256"],
             "redistributable": source_mode == "synthetic"},
        ],
        "random_seeds": [42 + s for s in range(args.seeds)],
        "environment": {
            "python": platform.python_version(),
            "pytorch": __import__("torch").__version__,
            "os": platform.platform(),
            "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        },
        "elapsed_seconds": elapsed,
        "status": "REPRODUCE_OK",
    }
    _validate(manifest, "manifest.schema.json", "manifest.json")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    (out_dir / "REPRODUCE_OK").write_text(
        f"reproduce_full OK source_mode={source_mode} elapsed={elapsed}s\n",
        encoding="utf-8")
    log.step(f"完成 (source_mode={source_mode}, elapsed={elapsed}s)")
    log.log(f">> REPRODUCE_OK -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
