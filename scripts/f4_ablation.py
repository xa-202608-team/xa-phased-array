# -*- coding: utf-8 -*-
"""F4-B2/B3: PA6 三类消融执行器（预注册: docs/f4_sensitivity_ablation_design.md §B2/§B3）。

变体数据 (从冻结基线 channel h5 派生, 独立目录):
  - b2_count   : x_ch[:,0] (p_drift_norm 连续退化量) ← 子阵存活占比 (subarray_features[...,6], 计数型)
  - b3_subagg  : 每子阵 x_ch ← 该轨迹 16 子阵逐窗均值 (破坏子阵分辨率)
  - b3_sparse  : 每通道序列 1/6 抽稀 (36h 等效 cadence; 标签绝对窗口数不变)

然后每变体: run_groups --groups ch_target_only_gru --seeds 3 (v2 口径, 无 k-shot),
≤3 并发独立进程 (独立 output-dir/jsonl, 不共享 executor)。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = sys.executable
BASE_CFG = ROOT / "configs" / "phased_array.yaml"
BASE_SIM = ROOT / "data/simulated/phased_array/sim_v2/seed_42/phased_array_all.h5"
BASE_CH = ROOT / "data/features/phased_array/schema_ch_v1/target/channel_features.h5"
OUT = ROOT / "outputs" / "f4_ablation"
VARIANTS = ["b2_count", "b3_subagg", "b3_sparse"]
ALIVE_IDX = 6          # subarray_features[..., 6] = 子阵存活通道占比
SPARSE_STEP = 6        # 1/6 抽稀 (6h -> 36h 等效 cadence)


def _copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def build_variant(name: str) -> Path:
    vdir = OUT / name
    vdir.mkdir(parents=True, exist_ok=True)
    ch_out = vdir / "channel_features.h5"
    if ch_out.exists():
        return ch_out
    need_sim = name == "b2_count"
    print(f"[{name}] 构建变体 h5...", flush=True)
    with h5py.File(BASE_CH, "r") as fin, \
         h5py.File(BASE_SIM, "r") as fsim, \
         h5py.File(ch_out, "w") as fout:
        _copy_attrs(fin, fout)
        for tk in sorted(fin.keys()):
            gin, gout = fin[tk], fout.create_group(tk)
            _copy_attrs(gin, gout)
            if need_sim:
                alive = fsim[tk]["subarray_features"][:, :, ALIVE_IDX]   # (T,16)
            subs = sorted(k for k in gin.keys() if k.startswith("sub_"))
            sub_lens = [int(gin[s]["x_ch"].shape[0]) for s in subs]
            # b3_subagg: 逐窗对"仍在遥测中的子阵"求均值 (失效子阵记录止于其 eol,
            # 聚合流物理语义 = 在轨子阵均值); 各子阵取聚合流在自身时程内的切片
            x_agg = None
            if name == "b3_subagg":
                Tmax = max(sub_lens)
                xs = np.full((len(subs), Tmax, 4), np.nan, dtype=np.float32)
                for i, s in enumerate(subs):
                    xs[i, : sub_lens[i]] = gin[s]["x_ch"][:]
                x_agg = np.nanmean(xs, axis=0).astype(np.float32)   # (Tmax,4)
            for sk, T_s in zip(subs, sub_lens):
                sin, sout = gin[sk], gout.create_group(sk)
                _copy_attrs(sin, sout)
                x = sin["x_ch"][:].astype(np.float32)
                if name == "b2_count":
                    x[:, 0] = alive[: x.shape[0], int(sin.attrs["sub_id"])]
                elif name == "b3_subagg":
                    x = x_agg[:T_s]
                elif name == "b3_sparse":
                    x = x[::SPARSE_STEP]
                sout.create_dataset("x_ch", data=x)
                for field in ("hi_ch", "rul_ch_windows", "rul_ch_norm"):
                    arr = sin[field][:]
                    if name == "b3_sparse":
                        arr = arr[::SPARSE_STEP]
                    sout.create_dataset(field, data=arr)
    # 变体 config: 仅改 channel_level.feature_path
    cfg = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    cfg["channel_level"]["feature_path"] = str(
        Path("outputs/f4_ablation") / name / "channel_features.h5")
    (vdir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[{name}] -> {ch_out}", flush=True)
    return ch_out


def run_variant(name: str, seeds: int) -> dict:
    vdir = OUT / name
    cfg_path = vdir / "config.yaml"
    # run_groups 产物名取 config stem ("config") → all_metrics_config.json;
    # Windows 收尾期 0xC0000409 退出码以产物哨兵判定 (同 reproduce_full 纪律)
    metrics = vdir / "all_metrics_config.json"
    if metrics.exists():
        return {"variant": name, "status": "already_done"}
    r = subprocess.run(
        [PY, "-u", "-m", "src.experiments.run_groups", "--config", str(cfg_path),
         "--seeds", str(seeds), "--level", "channel", "--groups", "ch_target_only_gru",
         "--output-dir", str(vdir)],
        cwd=ROOT, capture_output=True, text=True)
    log = vdir / "run.log"
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-3000:], encoding="utf-8")
    ok = metrics.exists()      # 产物哨兵判定 (0xC0000409 收尾期崩溃不作失败)
    return {"variant": name, "status": "ok" if ok else f"fail(rc={r.returncode})",
            "rc": r.returncode, "tail": (r.stdout or r.stderr)[-200:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--data-only", action="store_true", help="只构建变体 h5 不跑模型")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:                       # 数据构建快, 串行即可
        build_variant(name)
    if args.data_only:
        return 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda n: run_variant(n, args.seeds), VARIANTS))
    (OUT / "ablation_runs.json").write_text(
        __import__("json").dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = [r for r in results if not str(r["status"]).startswith(("ok", "already"))]
    for r in results:
        print(f">> {r['variant']}: {r['status']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
