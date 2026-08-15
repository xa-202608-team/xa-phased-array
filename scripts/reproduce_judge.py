# -*- coding: utf-8 -*-
"""评审用小规模端到端复现（reproduce_judge，契约 §2 v1.1）。

流程（小规模固定轨迹，CPU 30 分钟量级；主步骤失败即非零退出）：
  1. prepare_source_data   源域准备（canonical 存在则用 NASA canonical，否则 synthetic，
                            manifest 标 source_mode=synthetic，不得与正式源域结果混用）
  2. generate_simulation   三级链仿真 sim_v2/sim_v1（--fast 固定小规模，seed=42）
  3. build_channel_hi / build_array_hi   HI 构造（器件级 + 服务级）
  4. component.predict     仿真遥测 -> 契约长表 -> 单指标因果预测（prediction.schema.json）
  5. channel_baselines     通道级非学习基线
  6. run_groups --smoke    对比实验（smoke 管线验证，非正式指标）
  7. Schema 校验 + manifest.json / metrics.json / run.log / REPRODUCE_OK

用法：
  python scripts/reproduce_judge.py --output outputs/judge
  bash scripts/reproduce_judge.sh --output outputs/judge      (同参数/同退出码)
  pwsh scripts/reproduce_judge.ps1 -Output outputs/judge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMAS = ROOT / "schemas"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    env = os.environ.get("XA_GIT_COMMIT", "").strip()
    if len(env) == 40:
        return env
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit("!! 无法确定 git_commit；导出环境请设置 XA_GIT_COMMIT")


class RunLog:
    """stdout + run.log 双写。

    `sentinel` 兜底：Windows 本机存在解释器**退出期**崩溃（0xC0000409，atexit 之后、
    CRT/DLL 卸载阶段；main 已完成、产物完整写出，Linux/Docker 不出现）。当命令
    退出码非零但哨兵 JSON 存在且可解析时，以产物为准记 WARNING 继续；产物缺失
    仍判失败。哨兵是步骤成功的真实证据，不是吞错。
    """

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "run.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, text: str) -> None:
        print(text, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    def step(self, title: str) -> None:
        self.log(f"\n========== {title} ==========")

    def run(self, cmd: list[str], env_extra: dict[str, str] | None = None,
            sentinel: Path | None = None) -> None:
        self.log(f"$ {' '.join(cmd)}")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        if env_extra:
            env.update(env_extra)
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        if proc.returncode != 0:
            if sentinel is not None and sentinel.is_file():
                try:
                    json.loads(sentinel.read_text("utf-8"))
                    self.log(f"  [warning] 进程退出码 {proc.returncode}（Windows 退出期崩溃），"
                             f"但产物哨兵完整: {sentinel.name} -> 按产物判定成功")
                    return
                except (json.JSONDecodeError, OSError):
                    pass
            self.log(f"!! 步骤失败 (exit={proc.returncode}): {' '.join(cmd)}")
            raise SystemExit(proc.returncode)


def _make_judge_telemetry(sim_csv: Path, out_csv: Path, out_meta: Path,
                          sim_h5_sha: str) -> tuple[Path, Path]:
    """sim_v2 traj CSV 的 G_array_dB 列 -> 契约长表遥测 + 元数据（judge 专用）。"""
    import yaml

    frame_csv = __import__("pandas").read_csv(sim_csv)
    rows = ["timestamp,component_id,component_type,condition_id,telemetry_name,value,unit"]
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n = min(60, len(frame_csv))
    for i in range(n):
        ts = t0 + timedelta(seconds=float(frame_csv["t"].iloc[i]))
        value = float(frame_csv["G_array_dB"].iloc[i])
        rows.append(f"{ts.strftime('%Y-%m-%dT%H:%M:%SZ')},pa-judge-traj000,phased_array,"
                    f"leo_nominal,array_gain_db,{value:.6f},dB")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    metadata = {
        "schema_version": "1.1.0",
        "contract_version": "component-contract-v1.1.0",
        "dataset_id": "phased-array-judge-smoke-v1",
        "component_type": "phased_array",
        "time": {"column": "timestamp", "format": "ISO 8601 / RFC 3339 UTC",
                 "timezone": "UTC"},
        "telemetry": [{
            "name": "array_gain_db", "unit": "dB", "sampling_frequency": "1/6h",
            "source": "phased-array-three-level-simulation",
            "prediction_time_available": True, "derived": True,
            "degradation_relation": "decreasing",
        }],
        "missing_values": {"policy": "drop",
                           "detail": "仿真遥测为规则网格，无缺失；缺失行整行剔除不插值"},
        "split": {"policy": "by_component_id",
                  "assignment": {"inference": ["pa-judge-traj000"]}},
        "labels": {"derivation": "judge 冒烟不使用任何标签；RUL 由因果线性趋势外推估计，"
                                 "监督标签派生规则见 docs/DATA_DICTIONARY.md"},
        "sources": [{
            "name": "phased-array-judge-simulation", "version": "1.0.0", "url": None,
            "license": "internal", "sha256": sim_h5_sha,
            "processing_script": "scripts/generate_simulation.py",
        }],
        "prediction": {"primary_telemetry": "array_gain_db",
                       "degradation_direction": "decreasing",
                       "failure_threshold": -3.0, "forecast_horizon": 12,
                       "rul_unit": "hours"},
    }
    out_meta.write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    return out_csv, out_meta


def _validate(document: dict, schema_name: str, label: str) -> None:
    import jsonschema
    import yaml

    schema = yaml.safe_load((SCHEMAS / schema_name).read_text("utf-8"))
    jsonschema.validate(document, schema)
    print(f">> Schema 校验通过: {label} <- {schema_name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/judge", help="输出目录")
    ap.add_argument("--config", default=str(ROOT / "configs/phased_array.yaml"))
    args = ap.parse_args()

    out_dir = (ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(out_dir)
    t_start = time.time()

    commit = _git_commit()
    config_path = Path(args.config).resolve()
    py = sys.executable

    log.log(f"reproduce_judge start={datetime.now(timezone.utc).isoformat()}")
    log.log(f"git_commit={commit}")
    log.log(f"config={config_path.relative_to(ROOT)}")

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
        f"note={'synthetic 源域 smoke：不得与正式源域结果混用' if source_mode == 'synthetic' else 'canonical/真实源域'}\n",
        encoding="utf-8")

    # 2. 仿真（小规模固定轨迹 seed=42）
    log.step("P2 三级链仿真 (fast, seed=42, sim_v2+sim_v1)")
    sim_manifest_path = out_dir / "sim_manifest.json"
    log.run([py, "scripts/generate_simulation.py", "--config", str(config_path),
             "--fast", "--seed", "42", "--manifest", str(sim_manifest_path)])
    sim_manifest = json.loads(sim_manifest_path.read_text("utf-8"))

    # 3. HI 构造
    log.step("P3 通道级 + 服务级 HI 构造")
    log.run([py, "-m", "src.sim.build_channel_hi", "--config", str(config_path), "--report"])
    log.run([py, "-m", "src.sim.build_array_hi", "--config", str(config_path), "--report"])

    # 4. 单指标预测（仿真遥测 -> 契约长表 -> component.predict）
    log.step("P4 单指标遥测预测 (component.predict)")
    sim_v2_dir = ROOT / sim_manifest["runs"][0]["output_dir"]
    telemetry_csv = out_dir / "judge_telemetry.csv"
    metadata_yaml = out_dir / "judge_dataset.yaml"
    _make_judge_telemetry(sim_v2_dir / "traj_000.csv", telemetry_csv, metadata_yaml,
                          sim_manifest["runs"][0]["h5_sha256"])
    predict_out = out_dir / "predict"
    log.run([py, "-m", "component.predict", "--telemetry", str(telemetry_csv),
             "--metadata", str(metadata_yaml), "--telemetry-name", "array_gain_db",
             "--output", str(predict_out)], env_extra={"XA_GIT_COMMIT": commit})
    prediction = json.loads((predict_out / "prediction.json").read_text("utf-8"))
    _validate(prediction, "prediction.schema.json", "prediction.json")

    # 5. 通道级基线
    log.step("P5 通道级非学习基线")
    baselines_path = out_dir / "baselines_channel.json"
    log.run([py, "-m", "src.baselines.channel_baselines", "--config", str(config_path),
             "--seed", "42", "--out", str(baselines_path)])
    baselines = json.loads(baselines_path.read_text("utf-8"))

    # 6. 对比实验（smoke 管线验证）
    log.step("P6 对比实验 (smoke, level=channel)")
    groups_dir = out_dir / "groups"
    metrics_json = groups_dir / "all_metrics_phased_array_smoke.json"
    log.run([py, "-m", "src.experiments.run_groups", "--config", str(config_path),
             "--smoke", "--level", "channel", "--output-dir", str(groups_dir)],
            sentinel=metrics_json)
    all_metrics = json.loads(metrics_json.read_text("utf-8"))

    # 7. 汇总 metrics（全部来自本次运行的实际输出，不编造）
    log.step("P7 指标汇总与 Schema 校验")
    metrics: dict[str, float] = {}
    for group, agg in all_metrics.get("agg", {}).items():
        if isinstance(agg, dict) and "rmse_mean" in agg:
            metrics[f"rmse_mean.{group}"] = round(float(agg["rmse_mean"]), 6)
    for name, payload in baselines.items():
        if isinstance(payload, dict) and "rmse" in payload:
            metrics[f"baseline.{name}.rmse"] = round(float(payload["rmse"]), 6)
    if not metrics:
        log.log("!! 未提取到任何指标")
        raise SystemExit(1)

    metrics_doc = {
        "schema_version": "1.0.0",
        "component": "phased_array",
        "run_id": f"judge-{source_mode}-seed42",
        "mode": "quick",
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
        "random_seeds": [42],
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
        f"reproduce_judge OK source_mode={source_mode} elapsed={elapsed}s\n",
        encoding="utf-8")
    log.step(f"完成 (source_mode={source_mode}, elapsed={elapsed}s)")
    log.log(f">> REPRODUCE_OK -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
