# -*- coding: utf-8 -*-
"""F4-A: ±20% 物理参数敏感性 OAT 编排器（预注册: docs/f4_sensitivity_ablation_design.md §A）。

对 5 参数 × {−20%, +20%} 共 10 变体:
  1. 写变体 config (冻结 config 单键缩放, 其余逐字不动)
  2. 重生成 sim_v2 (n_traj=200, seed=42, --out 独立目录)
  3. build_channel_hi --indir/--out 独立路径
  4. 测量 (服务级 EOL/失效率/约束构成 + 通道级失效率/median EOL_ch + 物理自查)
     → outputs/f4_oat/<variant>/measure.json

冻结基线 (既有 sim_v2 + v2 channel h5) 不重跑, 同口径测量作对照行。
并行: ThreadPoolExecutor(max_workers=J) 管理独立子进程 (非共享 executor, 规避
Windows spawn 共享池硬崩教训, 见 progress F2 事故记录)。
"""
from __future__ import annotations

import argparse
import copy
import json
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
OUT_ROOT = ROOT / "outputs" / "f4_oat"
BASE_SIM = ROOT / "data/simulated/phased_array/sim_v2/seed_42/phased_array_all.h5"
BASE_CH = ROOT / "data/features/phased_array/schema_ch_v1/target/channel_features.h5"

# (variant, config 键路径, 缩放方式) — ±20% = 值×0.8 / ×1.2; 列表参数双端同缩
PARAMS = [
    ("ea", ["sim", "physics", "Ea_eV_range"]),
    ("thermal_cycle", ["sim", "physics", "damage_model", "coffin_manson", "deltaT_ref_K"]),
    ("rth_feedback", ["sim", "physics", "damage_model", "rth_feedback", "a5"]),
    ("life_sigma", ["sim", "physics", "life_scale_sigma"]),
    ("margin0", ["sim", "link_budget", "initial_margin_dB_range"]),
]


def _scale(val, factor):
    if isinstance(val, list):
        return [v * factor for v in val]
    return val * factor


def _get(cfg, keys):
    for k in keys:
        cfg = cfg[k]
    return cfg


def _set(cfg, keys, value):
    for k in keys[:-1]:
        cfg = cfg[k]
    cfg[keys[-1]] = value


def make_variant_cfg(direction: str) -> dict:
    """direction 如 'ea_minus' — 生成对应变体 config dict。"""
    name = direction.rsplit("_", 1)[0]
    sign = direction.rsplit("_", 1)[1]
    factor = 0.8 if sign == "minus" else 1.2
    cfg = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    keys = next(k for n, k in PARAMS if n == name)
    _set(cfg, keys, _scale(_get(cfg, keys), factor))
    return cfg


# ---------------------------------------------------------------- 测量
def _first_sustained(bad: np.ndarray, k: int) -> int | None:
    """首个连续 k 窗为 True 的起点; 无则 None。"""
    run = 0
    for i, b in enumerate(bad):
        run = run + 1 if b else 0
        if run >= k:
            return i - k + 1
    return None


def measure_sim(sim_h5: Path, cfg: dict) -> dict:
    """服务级统计: EOL_svc 中位 (绝对窗口)/失效率/约束构成 + 物理自查量。"""
    svc = cfg["sim"]["service_limits"]
    k = int(svc["consecutive_windows"])
    sll_max = float(svc["SLL_max_dB"])
    th_max = float(svc["theta_err_max_deg"])
    eols, binds, margins, eas = [], {"M_link": 0, "SLL": 0, "theta_err": 0}, [], []
    n_censored = 0
    T_ref = None
    with h5py.File(sim_h5, "r") as f:
        for tk in sorted(f.keys()):
            g = f[tk]
            m = g["M_link_dB_true"][:]
            s = g["SLL_dB_true"][:]
            th = np.abs(g["theta_err_deg_true"][:])
            T_ref = len(m)
            bad_link = m <= 0.0
            bad_sll = s > sll_max
            bad_th = th > th_max
            e_link, e_sll, e_th = (_first_sustained(b, k) for b in (bad_link, bad_sll, bad_th))
            cands = [(e, "M_link") for e in [e_link] if e is not None] + \
                    [(e, "SLL") for e in [e_sll] if e is not None] + \
                    [(e, "theta_err") for e in [e_th] if e is not None]
            if cands:
                eol, bind = min(cands, key=lambda x: x[0])
                eols.append(eol)
                binds[bind] += 1
            else:
                n_censored += 1
            margins.append(float(g.attrs["margin0_dB"]))
            eas.append(float(g.attrs["Ea_eV"]))
    n = len(eols) + n_censored
    return {
        "n_traj": n,
        "svc_failure_rate": len(eols) / n,
        "svc_eol_median_windows": float(np.median(eols)) if eols else None,
        "svc_binding": binds,
        "n_censored": n_censored,
        "horizon_windows": T_ref,
        "median_margin0_dB": float(np.median(margins)),
        "median_Ea_eV": float(np.median(eas)),
    }


def measure_ch(ch_h5: Path) -> dict:
    """通道级统计: 失效率 + 失效通道 median EOL_ch (绝对窗口)。"""
    sys.path.insert(0, str(ROOT))
    from src.transfer.channel_dataset import read_channel_label_meta
    fails, cens = [], 0
    with h5py.File(ch_h5, "r") as f:
        meta = read_channel_label_meta(f)
        for tk in sorted(f.keys()):
            for sk in sorted(k2 for k2 in f[tk].keys() if k2.startswith("sub_")):
                sub = f[tk][sk]
                if bool(sub.attrs["event_observed"]):
                    fails.append(float(sub["rul_ch_windows"][0]))
                else:
                    cens += 1
    n = len(fails) + cens
    return {
        "n_channels": n,
        "ch_failure_rate": len(fails) / n,
        "ch_eol_median_windows_failed_only": float(np.median(fails)) if fails else None,
        "meta": {"H": meta["rul_scale_windows"], "sp_s": meta["sample_period_s"]},
    }


# ---------------------------------------------------------------- 单变体执行
def run_variant(direction: str, n_traj: int) -> dict:
    vdir = OUT_ROOT / direction
    vdir.mkdir(parents=True, exist_ok=True)
    sim_dir = vdir / "sim"
    ch_h5 = vdir / "channel_features.h5"
    cfg_path = vdir / "config.yaml"
    if not cfg_path.exists():
        cfg_path.write_text(
            yaml.safe_dump(make_variant_cfg(direction), allow_unicode=True, sort_keys=False),
            encoding="utf-8")
    if not (sim_dir / "phased_array_all.h5").exists():
        print(f"[{direction}] sim 生成中 (n_traj={n_traj})...", flush=True)
        r = subprocess.run([PY, "-m", "src.sim.phased_array_sim", "--config", str(cfg_path),
                            "--n_traj", str(n_traj), "--seed", "42", "--out", str(sim_dir)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            return {"variant": direction, "error": f"sim 失败: {r.stderr[-500:]}"}
    if not ch_h5.exists():
        print(f"[{direction}] build_channel_hi...", flush=True)
        r = subprocess.run([PY, "-m", "src.sim.build_channel_hi", "--config", str(BASE_CFG),
                            "--in", str(sim_dir), "--out", str(ch_h5)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            return {"variant": direction, "error": f"HI 失败: {r.stderr[-500:]}"}
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = {"variant": direction}
    out.update(measure_sim(sim_dir / "phased_array_all.h5", cfg))
    out.update(measure_ch(ch_h5))
    (vdir / "measure.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{direction}] 完成: svc_fail={out['svc_failure_rate']:.3f} "
          f"ch_fail={out['ch_failure_rate']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=3, help="sim 并发子进程数")
    ap.add_argument("--skip-variants", action="store_true",
                    help="只重测基线与已有变体 (不跑新 sim)")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    base = {"variant": "baseline(frozen)"}
    base.update(measure_sim(BASE_SIM, base_cfg))
    base.update(measure_ch(BASE_CH))
    (OUT_ROOT / "baseline_measure.json").write_text(
        json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[baseline] svc_fail={:.3f} ch_fail={:.3f}".format(
        base["svc_failure_rate"], base["ch_failure_rate"]), flush=True)

    directions = [f"{n}_{s}" for n, _ in PARAMS for s in ("minus", "plus")]
    if args.skip_variants:
        results = [run_variant(d, 0) for d in directions
                   if (OUT_ROOT / d / "channel_features.h5").exists()]
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(lambda d: run_variant(d, args.n_traj), directions))

    (OUT_ROOT / "oat_results.json").write_text(
        json.dumps({"baseline": base, "variants": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    bad = [r for r in results if "error" in r]
    print(f">> OAT 完成: {len(results) - len(bad)}/{len(results)} 变体成功, "
          f"结果 -> {OUT_ROOT / 'oat_results.json'}")
    if bad:
        for r in bad:
            print(f"!! {r['variant']}: {r['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
