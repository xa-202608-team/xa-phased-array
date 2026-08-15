"""baselines/physical_extrap.py

物理基线 (plan §四 Phase 6): 滑窗辨识 b̂ → 线性外推到 b_fail → RUL。
合理物理经验方法, 作为非黑箱对照。

评估指标 (全项目共享):
  RMSE       均方根误差
  PHM Score  晚预测重罚 (PHM2008: d=pred-true, d>0 用 exp(d/10), d<0 用 exp(-d/13))
  MAE        平均绝对误差

用法:
  python -m src.baselines.physical_extrap --config configs/phased_array.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed            # noqa: E402


# ---------------------------------------------------------------- 指标
def rmse(pred, true) -> float:
    pred, true = np.asarray(pred), np.asarray(true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def phm_score(pred, true) -> float:
    """PHM2008 score: 晚预测 (pred>true) 指数重罚。"""
    d = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    d = np.clip(d, -50.0, 50.0)   # 防 exp 溢出 (大误差时 PHM 指数爆 inf, 如 hi_extrap 在非线性退化下)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


# ---------------------------------------------------------------- 物理外推
def physical_extrap_rul(b_hat, t_idx, b_fail, window: int = 50) -> np.ndarray:
    """滑窗线性拟合 b̂ 趋势, 外推到 b_fail 估计 RUL。

    对每个 t, 用 [t-window, t] 的 b̂ 线性拟合 (slope, intercept),
    外推失效时刻 t_fail=(b_fail-intercept)/slope, RUL=t_fail-t。
    """
    n = len(b_hat)
    rul = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window)
        ti = t_idx[lo:i + 1]
        bi = b_hat[lo:i + 1]
        if len(ti) < 5:
            continue
        slope, intercept = np.polyfit(ti, bi, 1)
        if slope <= 1e-12:
            continue
        t_fail = (b_fail - intercept) / slope
        rul[i] = max(0.0, t_fail - t_idx[i])
    return rul


def evaluate_physical(cfg: dict, seed: int) -> dict | None:
    set_seed(seed, cfg["reproducibility"]["deterministic"])
    target_h5 = ROOT / cfg["transfer"].get("target_feature_path", "data/features/wheel/schema_v1/target_features.h5")
    if not target_h5.exists():
        print(f"!! 缺 {target_h5}; 先 python -m src.sim.build_hi --report")
        return None
    rmses, scores, maes = [], [], []
    with h5py.File(target_h5, "r") as f:
        traj_keys = sorted(f.keys())
        if not traj_keys:
            return None
        # 字段保护: 相控阵 target (x_global/hi_array) 无 b_hat, Arrhenius 基线待 PA6 后续
        if "b_hat" not in f[traj_keys[0]]:
            print(f">> 物理外推基线: {target_h5} 无 b_hat 字段 (相控阵 target 用 x_global/hi_array), "
                  f"跳过 (Arrhenius 物理基线待 PA6 后续实现)")
            return None
        for k in traj_keys:
            g = f[k]
            b_hat = uniform_filter1d(g["b_hat"][:], 30)
            b_true = g["b_true"][:]
            rul_abs = g["rul"][:].astype(float)
            lf = g["label_fail"][:]
            eol = int(np.argmax(lf)) if lf.any() else len(b_hat) - 1
            b_fail = b_true[eol]
            rul_max = max(float(rul_abs.max()), 1.0)
            rul_true_n = rul_abs / rul_max
            t_idx = np.arange(len(b_hat))
            rul_est = physical_extrap_rul(b_hat, t_idx, b_fail, 50) / rul_max
            valid = ~np.isnan(rul_est) & (t_idx < eol)
            if int(valid.sum()) < 5:
                continue
            re = rul_est[valid]
            rt = rul_true_n[valid]
            rmses.append(rmse(re, rt))
            scores.append(phm_score(re, rt))
            maes.append(mae(re, rt))
    return {"rmse": float(np.mean(rmses)),
            "phm": float(np.mean(scores)),
            "mae": float(np.mean(maes)),
            "n_traj": len(rmses)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="checkpoints/physical_metrics.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    m = evaluate_physical(cfg, seed)
    if m:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f">> physical baseline: RMSE={m['rmse']:.4f} PHM={m['phm']:.2f} "
              f"MAE={m['mae']:.4f} ({m['n_traj']} traj) -> {out}")


if __name__ == "__main__":
    main()
