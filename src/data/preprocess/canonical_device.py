"""src/data/preprocess/canonical_device.py

Canonical device schema (4 维) — 源域 NASA MOSFET 与目标域 GaN 子阵同口径。

迁移核心: 两域第0维 p_drift_norm 都是"归一化到各自器件失效阈值的关键参量漂移",
语义一致 (源 ΔRDS/δR_src, 目标 max(ΔIDSS,ΔP)/δI)。第1-3维是共享应力/工况协变量。

T3 (M3): 源域 (加速老化台架, 密集测量, 时间尺度压缩) 与目标域 (6h 在轨遥测窗)
统一重采样到 n_win=512 归一化寿命进度点 — 在数据侧解决异构, 不留给网络
(旧方案 adapter 12→2 瓶颈的直接教训)。
"""
from __future__ import annotations

import numpy as np

CANONICAL_SCHEMA = "device_canonical_v1"
CANONICAL_COLS = ["p_drift_norm", "T_dev_C", "duty", "drive_norm"]
CANONICAL_N_FEATURES = 4

# schema_v3 源域列名 (mosfet_source_features.h5 file attrs feature_names)
V3_COLS = ["RDS_drift", "T_case_C", "supply_V", "gate_voltage", "duty_cycle"]


def to_canonical_source(features: np.ndarray, delta_R_src: float = 0.05,
                        feature_names: list[str] | None = None) -> np.ndarray:
    """schema_v3 源域 features (N,5) → canonical (N,4)。

    列映射 (默认 V3_COLS 顺序, 可由 h5 attrs feature_names 覆盖):
      p_drift_norm = RDS_drift / delta_R_src   (归一化到 Si MOSFET 失效阈值, NASA Celaya 0.05)
      T_dev_C      = T_case_C                  (器件壳温)
      duty         = duty_cycle                (工况占空比)
      drive_norm   = supply_V * gate_voltage, 归一到该器件初值 (驱动强度协变量)
    """
    cols = list(feature_names) if feature_names else V3_COLS
    idx = {c: i for i, c in enumerate(cols)}
    RDS = np.asarray(features[:, idx["RDS_drift"]], dtype=np.float64)
    T_case = np.asarray(features[:, idx["T_case_C"]], dtype=np.float64)
    supply = np.asarray(features[:, idx["supply_V"]], dtype=np.float64)
    gate = np.asarray(features[:, idx["gate_voltage"]], dtype=np.float64)
    duty = np.asarray(features[:, idx["duty_cycle"]], dtype=np.float64)
    p_drift = RDS / max(float(delta_R_src), 1e-9)
    drive = supply * gate
    drive0 = abs(drive[0]) if len(drive) and abs(drive[0]) > 1e-9 else 1.0
    drive_norm = drive / drive0
    return np.stack([p_drift, T_case, duty, drive_norm], axis=1).astype(np.float32)


def to_canonical_target(sa_feat_s: np.ndarray, duty: float, deltas: dict) -> np.ndarray:
    """子阵 sa_feat (T,8) → canonical (T,4)。复用 build_channel_hi.build_canonical_x 口径。"""
    from src.sim.build_channel_hi import build_canonical_x
    return build_canonical_x(sa_feat_s, duty, deltas)


def resample_to_windows(x: np.ndarray, t_s: np.ndarray, n_win: int = 512) -> np.ndarray:
    """按归一化寿命进度等间隔重采样到 n_win 点。

    t_s: 时间戳 (T,)。用观测时长归一化到 [0,1] (删失安全: t_s[-1]=观测截止时刻,
    非真 EOL, 无未来信息泄漏)。每列线性插值。返回 (n_win, F) float32。
    """
    T = len(x)
    F = x.shape[1] if x.ndim > 1 else 1
    if T == 0:
        return np.zeros((n_win, F), dtype=np.float32)
    x2 = np.asarray(x, dtype=np.float64).reshape(T, F)
    if T == 1:
        return np.repeat(x2, n_win, axis=0).astype(np.float32)
    t = np.asarray(t_s, dtype=np.float64)
    span = float(t[-1] - t[0])
    if span < 1e-9:
        return np.repeat(x2, n_win, axis=0).astype(np.float32)
    t_norm = (t - t[0]) / span
    grid = np.linspace(0.0, 1.0, n_win)
    out = np.empty((n_win, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(grid, t_norm, x2[:, f])
    return out
