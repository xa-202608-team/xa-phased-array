"""src/metrics/prognostic.py

提前性指标 (T5/M5): PH / α-λ / CM / RA — 赛题第六节"具备一定提前性"。

标准 PHM 定义 (Saxena et al. 2008; 赛题要求 PH 与 α-λ 进主结果表):
  alpha_lambda_accuracy: 寿命进度 λ 处预测是否落入 [(1-α)RUL*, (1+α)RUL*] (二值)
  prognostic_horizon:   首次进入 α 带并保持至 EOL 的时刻到 EOL 的距离 (提前量, 窗;
                        完美预测 → PH = EOL; 恒定偏置 >α → PH = 0)
  convergence_metric:   α 带外误差面积的重心距 EOL 的距离 (越小越早收敛)
  relative_accuracy:    1 - |pred-true|/true (寿命进度 λ 处; 越接近 1 越准)

输入 rul_pred/rul_true 为 (T,) 数组, 单位"窗"(6h); EOL = RUL 首次 ≤0 的索引。
"""
from __future__ import annotations

import numpy as np


def _eol_idx(rul_true: np.ndarray) -> int:
    """RUL 首次 ≤ 0 的索引 (EOL); 无则末尾。"""
    rul_true = np.asarray(rul_true, dtype=np.float64)
    hit = np.where(rul_true <= 0)[0]
    return int(hit[0]) if hit.size else len(rul_true) - 1


def alpha_lambda_accuracy(rul_pred, rul_true, alpha: float = 0.2, lam: float = 0.5) -> bool:
    """寿命进度 λ 处预测是否落入 [(1-α)RUL*, (1+α)RUL*]。"""
    rp = np.asarray(rul_pred, dtype=np.float64)
    rt = np.asarray(rul_true, dtype=np.float64)
    eol = _eol_idx(rt)
    t = min(int(lam * eol), eol)
    lo = (1.0 - alpha) * rt[t]
    hi = (1.0 + alpha) * rt[t]
    return bool(lo <= rp[t] <= hi)


def prognostic_horizon(rul_pred, rul_true, alpha: float = 0.2) -> int:
    """首次进入 α 带并保持至 EOL 的时刻到 EOL 的距离 (提前量, 窗)。

    从 EOL 往前扫, 找最晚的"断裂点"(outside), 其后即为持续 inside 段;
    PH = EOL - (断裂点+1)。完美预测 PH=EOL; 全程 outside PH=0。
    """
    rp = np.asarray(rul_pred, dtype=np.float64)
    rt = np.asarray(rul_true, dtype=np.float64)
    eol = _eol_idx(rt)
    if eol <= 0:
        return 0
    lo = (1.0 - alpha) * rt[:eol]
    hi = (1.0 + alpha) * rt[:eol]
    inside = (rp[:eol] >= lo) & (rp[:eol] <= hi)
    # 从 eol-1 往前找连续 inside 的最早起点 (持续至 EOL)
    first_hold = eol
    for t in range(eol - 1, -1, -1):
        if inside[t]:
            first_hold = t
        else:
            break
    return eol - first_hold


def convergence_metric(rul_pred, rul_true, alpha: float = 0.2) -> float:
    """α 带外误差面积的重心距 EOL 的距离 (越小越早收敛; 0=全程在带内)。"""
    rp = np.asarray(rul_pred, dtype=np.float64)
    rt = np.asarray(rul_true, dtype=np.float64)
    eol = _eol_idx(rt)
    if eol <= 0:
        return 0.0
    lo = (1.0 - alpha) * rt[:eol]
    hi = (1.0 + alpha) * rt[:eol]
    outside = ~((rp[:eol] >= lo) & (rp[:eol] <= hi))
    if not outside.any():
        return 0.0
    err = np.abs(rp[:eol] - rt[:eol])
    total = float(np.sum(err[outside]))
    if total < 1e-12:
        return 0.0
    centroid = float(np.sum(np.arange(eol)[outside] * err[outside]) / total)
    return float(eol - centroid)


def relative_accuracy(rul_pred, rul_true, lam: float = 0.5) -> float:
    """寿命进度 λ 处相对精度: 1 - |pred-true|/true (越接近 1 越准)。"""
    rp = np.asarray(rul_pred, dtype=np.float64)
    rt = np.asarray(rul_true, dtype=np.float64)
    eol = _eol_idx(rt)
    t = min(int(lam * eol), eol)
    denom = abs(rt[t])
    if denom < 1e-9:
        return 0.0
    return float(1.0 - abs(rp[t] - rt[t]) / denom)
