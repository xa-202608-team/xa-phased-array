"""baselines/channel_baselines.py

通道级 5 个对比基线 (T7/M6, channel_level 路线)。

赛题第六节要求"与至少一种现有寿命预测或退化建模方法进行对比实验, 在相同数据条件下
对预测误差、稳定性或提前性等指标进行分析"。本模块 5 个基线覆盖四类范式:

| 基线 | 说明 | 范式 |
|------|------|------|
| constant            | ŷ ≡ mean(RUL_train)               | 非学习下界 |
| z_extrap            | 子阵 z 滑窗线性外推到 z=1         | 物理启发 (非学习) |
| arrhenius_dose      | 遥测 Tj,duty 积分剂量外推(标称 Ea) | 纯物理机理 |
| similarity_matching | train 失效退化轨迹库 DTW + 加权 RUL | 现有方法 ① |
| particle_filter     | 状态 [z, ż], 量测 p_drift_norm, SIR | 现有方法 ② |

评估口径与主模型 run_groups 完全一致:
  - 轨迹级 split (同 traj_id 的 16 子阵同属一 split, 禁止打散)
  - rul_max_norm 固定物理上限归一 (跨 seed 可比, 禁 train-only max)
  - 仅失效通道算 RMSE/PHM/MAE; 删失通道报下界违反率
    (与 eval_test 同口径: 删失无精确 RUL, 混算会把"距仿真截止时刻"当退化标签)

用法:
  python -m src.baselines.channel_baselines --config configs/phased_array.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                                   # noqa: E402
from src.baselines.physical_extrap import rmse, mae, phm_score                # noqa: E402
from src.transfer.train_transfer import split_trajectories                    # noqa: E402

K_BOLTZMANN_eVperK = 8.617333262e-5

# canonical device schema (build_channel_hi.CANONICAL_COLS) 列序
COL_P_DRIFT = 0     # 归一化到器件失效阈值的关键参量漂移 (源/目标同语义, 迁移核心)
COL_T_DEV = 1       # T_dev_C (器件壳温 / 结温, Arrhenius 输入)
COL_DUTY = 2        # duty (工况占空比)
COL_DRIVE = 3       # drive_norm (驱动强度协变量)


# ---------------------------------------------------------------- 数据加载
def _load_channels(h5_path: Path) -> list[dict]:
    """读 channel_features.h5 → list of dict (每子阵一个通道记录)。

    每条记录:
      x_ch (T,4), hi_ch (T,), z_ch (T,), rul_ch (T,) 单位为**绝对窗口数**,
      event, eol_idx, traj_id, sub_id, traj_attrs (轨迹级物理参数)

    F1-A: 按 h5 schema 读字段 — v2 (channel_label_v2) 读 rul_ch_windows (不归一),
    v1 (legacy) 读 rul_ch (已 cap 的窗口数)。调用方按 rul_max (H 或 4088) 自行归一,
    杜绝把 rul_ch_norm 再除一次 4088 的双归一。
    """
    from src.transfer.channel_dataset import (
        read_channel_label_meta, CHANNEL_LABEL_SCHEMA_V2)
    channels: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        meta = read_channel_label_meta(f)
        rul_field = ("rul_ch_windows" if meta["channel_label_schema"] == CHANNEL_LABEL_SCHEMA_V2
                     else "rul_ch")
        for key in sorted(f.keys()):
            traj = f[key]
            attrs = {k: (float(v) if np.isscalar(v) else np.asarray(v))
                     for k, v in traj.attrs.items()}
            for sk in sorted(k for k in traj.keys() if k.startswith("sub_")):
                sub = traj[sk]
                channels.append(dict(
                    x_ch=sub["x_ch"][:].astype(float),
                    hi_ch=sub["hi_ch"][:].astype(float),
                    z_ch=sub["z_ch"][:].astype(float),
                    rul_ch=sub[rul_field][:].astype(float),
                    event=bool(int(sub.attrs["event_observed"])),
                    eol_idx=int(sub.attrs["eol_idx"]),
                    traj_id=int(sub.attrs["traj_id"]),
                    sub_id=int(sub.attrs["sub_id"]),
                    traj_attrs=attrs,
                ))
    return channels


# ---------------------------------------------------------------- 基线 1: constant
def _constant_predict(rul_test_norm: np.ndarray, const_val: float) -> np.ndarray:
    """所有 test 点预测同一常数 (train 集失效通道 rul 归一均值)。

    愚蠢下界: 若主模型 RMSE 贴近此值, 说明模型没在学退化动力学。
    """
    return np.full_like(rul_test_norm, const_val, dtype=float)


# ---------------------------------------------------------------- 基线 2: z 线性外推
def _z_extrap_predict(z_ch: np.ndarray, rul_max: float, window: int = 50) -> np.ndarray:
    """子阵 z 滑窗线性外推到 z=1.0 估计 RUL。

    对每个 t, 用 [t-window, t] 的 z 线性拟合 (slope, intercept),
    外推失效时刻 t_fail=(1.0-intercept)/slope, RUL=t_fail-t。
    早期 (<window) 或 slope<=0 时返回 NaN, 由调用方用常数兜底 (保所有点可比)。

    z=max(dR/δR,dI/δI,dg/δg) 单调累积 → 线性外推是物理启发的非学习基线。
    """
    n = len(z_ch)
    rul = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window)
        seg = z_ch[lo:i + 1]
        if len(seg) < 5:
            continue
        slope, intercept = np.polyfit(np.arange(len(seg)), seg, 1)
        if slope <= 1e-9:                 # z 不上升 (健康早期), 无法外推
            continue
        t_fail = (1.0 - intercept) / slope + lo
        rul[i] = max(0.0, t_fail - i)
    return rul / max(rul_max, 1.0)


# ---------------------------------------------------------------- 基线 3: Arrhenius 剂量
def _arrhenius_rate(Tj_K: np.ndarray, Ea_eV: float) -> np.ndarray:
    """Arrhenius 反应速率 r = exp(-Ea/(k_B·Tj)) (相对量, dt=1 窗)。"""
    return np.exp(-Ea_eV / (K_BOLTZMANN_eVperK * Tj_K))


def _arrhenius_predict(Tj_K: np.ndarray, D_eol: float, rul_max: float,
                       Ea_eV: float) -> np.ndarray:
    """Arrhenius 剂量积分外推: RUL=(D_EOL-D_now)/rate_now (剩余剂量/当前剂量率)。

    D(t)=Σ exp(-Ea/(k_B·Tj)); D_EOL 由 train 失效通道 EOL 处剂量中位标定。
    纯物理机理基线 (标称 Ea, 不用轨迹真值 Ea_eV; CLAUDE.md 第 9 条硬约束)。
    D_now>=D_EOL (已超 train 标定失效) 时 clip 0。
    """
    rate = _arrhenius_rate(Tj_K, Ea_eV)
    D = np.cumsum(rate)
    remaining = (D_eol - D) / np.maximum(rate, 1e-12)
    return np.clip(remaining, 0.0, None) / max(rul_max, 1.0)


# ---------------------------------------------------------------- 基线 4: 相似性匹配
def _dtw_distance(q: np.ndarray, r: np.ndarray, sakoe_frac: float = 0.25) -> float:
    """单维序列 DTW 距离 (Sakoe-Chiba 带宽约束, |i-j|<=sakoe_frac·m)。

    输入长度应已重采样到统一长度 (如 48), 避免算力爆炸。
    带宽约束把内层循环从 O(m) 降到 O(sakoe_frac·m), 配合 numpy 切片向量化。
    """
    n, m = len(q), len(r)
    if n == 0 or m == 0:
        return float("inf")
    sakoe = max(1, int(m * sakoe_frac))
    INF = float("inf")
    D = np.full(m + 1, INF)
    D[0] = 0.0
    for i in range(1, n + 1):
        jlo = max(1, i - sakoe)
        jhi = min(m, i + sakoe)
        cost_row = np.abs(q[i - 1] - r[jlo - 1:jhi])   # (jhi-jlo+1,)
        new_D = np.full(m + 1, INF)
        up = D[jlo:jhi + 1]                            # D[i-1, jlo..jhi]
        diag = D[jlo - 1:jhi]                          # D[i-1, jlo-1..jhi-1]
        # 顺序循环处理左方依赖 (D[i, j-1]); 带宽内 jhi-jlo+1 步
        prev = INF                                      # D[i, jlo-1] 边界
        for k in range(jhi - jlo + 1):
            cur = cost_row[k] + min(prev, up[k], diag[k])
            new_D[jlo + k] = cur
            prev = cur
        D = new_D
    return float(D[m])


def _resample_curve(curve: np.ndarray, n_grid: int = 64) -> np.ndarray:
    """把不等长曲线等距重采样到 n_grid 点 (DTW 前置统一长度)。"""
    T = len(curve)
    if T == 0:
        return np.zeros(n_grid)
    if T == 1:
        return np.full(n_grid, curve[0])
    idx = np.linspace(0, T - 1, n_grid)
    lo = idx.astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = idx - lo
    return curve[lo] * (1 - frac) + curve[hi] * frac


def _similarity_predict(test_curve: np.ndarray, library: list[tuple],
                        rul_max: float, top_k: int = 5,
                        n_grid: int = 32, n_anchors: int = 20) -> np.ndarray:
    """与 train 失效通道轨迹库做 DTW 匹配 + 加权 RUL。

    library: list of (p_drift_curve, rul_remaining_curve), 来自 train 失效通道
    对 test 通道当前段 test_curve[:t+1]:
      1. 重采样到 n_grid 点
      2. 与 library 中每条 p_drift_curve (同样重采样到 n_grid) 做 DTW
      3. top-k 最近邻按 1/(d+eps) 加权, 取各自"对齐位置"处的 remaining 平均作 RUL

    性能: 只在 n_anchors 个锚点处重算 (其余线性插值)。
    对齐位置: 由 test_curve[t] 的值映射到 library curve 上的最近 index (单调近似)。
    """
    n = len(test_curve)
    if not library or n < 5:
        return np.full(n, np.nan)
    preds = np.full(n, np.nan)
    # 预计算参考曲线重采样
    ref_resampled = [_resample_curve(c[0], n_grid) for c in library]
    ref_remaining = [c[1] for c in library]
    ref_orig = [c[0] for c in library]
    anchors = np.linspace(5, n - 1, n_anchors).astype(int)
    for t in anchors:
        q = _resample_curve(test_curve[:t + 1], n_grid)
        dists = np.asarray([_dtw_distance(q, r) for r in ref_resampled])
        top = np.argsort(dists)[:top_k]
        weights = 1.0 / (dists[top] + 1e-9)
        rem_estimates = []
        target_val = test_curve[t]
        for idx in top:
            curve = ref_orig[idx]
            remaining = ref_remaining[idx]
            pos = int(np.argmin(np.abs(curve - target_val)))
            rem_estimates.append(float(remaining[pos]))
        rem_estimates = np.asarray(rem_estimates)
        preds[t] = float(np.sum(weights * rem_estimates) / np.sum(weights)) / max(rul_max, 1.0)
    # 线性插值填充锚点之间 + 前向后向填充
    valid = np.isfinite(preds)
    if valid.any():
        idx = np.arange(n)
        preds = np.interp(idx, idx[valid], preds[valid])
        # 前 5 点用首个有限值填充
        first_valid = int(np.argmax(valid))
        preds[:first_valid] = preds[first_valid]
    return preds


# ---------------------------------------------------------------- 基线 5: 粒子滤波
def _particle_filter_predict(p_drift: np.ndarray, rul_max: float,
                             n_particles: int = 200, seed: int = 42,
                             process_sigma_z: float = 5e-3,
                             process_sigma_rate: float = 5e-4,
                             meas_sigma: float = 2e-2) -> np.ndarray:
    """PF 估计状态 [z, ż], 量测 p_drift≈z+noise, RUL=(1-z)/ż。

    标准 SIR (Sampling Importance Resampling):
      过程 (随机游走):
        z_{t+1}   = max(0, z_t + rate_t·dt + σ_z·N(0,1))
        rate_{t+1} = max(eps, rate_t + σ_rate·N(0,1))
      量测:
        p_drift_t = z_t + σ_meas·N(0,1)   (canonical 第0维 ≈ z + 噪声)
      RUL(t):
        z 加权均值 z̄, rate 加权均值 ṙ → RUL=max(0, (1-z̄)/ṙ)
      重采样:
        有效粒子数 N_eff < N/2 时系统重采样

    失效阈值 z=1 (与 build_channel_labels z=max(...)/δ 一致)。
    """
    rng = np.random.default_rng(seed)
    T = len(p_drift)
    # 初始化: z 从 [0, 0.05], rate 从 [0, 0.001]
    z = rng.uniform(0, 0.05, n_particles)
    rate = rng.uniform(0, 1e-3, n_particles)
    w = np.full(n_particles, 1.0 / n_particles)
    preds = np.zeros(T)
    for t in range(T):
        # ---- 过程更新 ----
        z = z + rate
        rate = np.maximum(rate + rng.normal(0, process_sigma_rate, n_particles), 1e-12)
        z = np.maximum(z + rng.normal(0, process_sigma_z, n_particles), 0.0)
        # ---- 量测更新 (p_drift ≈ z + noise) ----
        meas = p_drift[t]
        lik = np.exp(-0.5 * ((z - meas) / meas_sigma) ** 2) + 1e-300
        w = w * lik
        w_sum = w.sum()
        if w_sum < 1e-300:
            # 全粒子坍缩 → 重置均匀 (鲁棒兜底)
            w = np.full(n_particles, 1.0 / n_particles)
        else:
            w = w / w_sum
        # ---- RUL 估计 (加权粒子均值) ----
        z_mean = float(np.sum(w * z))
        rate_mean = float(np.sum(w * rate))
        preds[t] = max(0.0, (1.0 - z_mean) / max(rate_mean, 1e-12))
        # ---- 重采样 (N_eff < N/2) ----
        neff = 1.0 / float(np.sum(w * w))
        if neff < n_particles / 2:
            cum = np.cumsum(w)
            u = (rng.uniform(0, 1, n_particles) + np.arange(n_particles)) / n_particles
            idx = np.searchsorted(cum, u).clip(0, n_particles - 1)
            z = z[idx]
            rate = rate[idx]
            w = np.full(n_particles, 1.0 / n_particles)
    return preds / max(rul_max, 1.0)


# ---------------------------------------------------------------- 评估
def _evaluate_split(true_norm: np.ndarray, pred_norm: np.ndarray,
                    event_mask: np.ndarray) -> dict:
    """同 run_groups.eval_test: 仅失效通道算 RMSE/PHM/MAE; 删失报下界违反率。

    删失无精确 RUL, 混算会把"距仿真截止时刻"当退化标签 (复核第三条铁律)。
    """
    m = event_mask
    cm = ~m
    rmse_f = float(np.sqrt(np.mean((pred_norm[m] - true_norm[m]) ** 2))) if m.any() else 0.0
    mae_f = float(np.mean(np.abs(pred_norm[m] - true_norm[m]))) if m.any() else 0.0
    censor_viol = float(np.mean(pred_norm[cm] < true_norm[cm])) if cm.any() else 0.0
    return {
        "rmse": rmse_f, "mae": mae_f,
        "phm": phm_score(pred_norm[m], true_norm[m]) if m.any() else 0.0,
        "censor_violation_rate": censor_viol,
        "n_failed": int(m.sum()), "n_censored": int(cm.sum()),
    }


# ---------------------------------------------------------------- 主评估入口
def evaluate_channel_baselines(cfg: dict, seed: int, verbose: bool = True) -> dict | None:
    """跑 5 个基线, 返回各基线指标字典 + 协议字段。

    划分与 run_groups 完全一致 (按 traj_id), rul 按 h5 尺度元数据归一:
    v2 → H (rul_scale_windows, 任务视界, 与模型标签同口径); v1 → transfer.rul_max_norm。
    评估仅 test 失效通道算 RMSE/PHM/MAE, 删失通道报下界违反率。
    """
    set_seed(seed, cfg["reproducibility"]["deterministic"])
    ch = cfg["channel_level"]
    tcfg = cfg["transfer"]
    target_h5 = ROOT / ch["feature_path"]
    if not target_h5.exists():
        print(f"!! 缺 {target_h5}; 先 python -m src.sim.build_channel_hi --report")
        return None
    # F1-A: rul_max 从 h5 元数据读 (build_channel_hi 唯一计算者), 不再写死 4088
    from src.transfer.channel_dataset import (
        read_channel_label_meta, CHANNEL_LABEL_SCHEMA_V2)
    with h5py.File(target_h5, "r") as _f:
        _meta = read_channel_label_meta(_f)
    if _meta["channel_label_schema"] == CHANNEL_LABEL_SCHEMA_V2:
        rul_max = float(_meta["rul_scale_windows"])
    else:
        rul_max = float(tcfg["rul_max_norm"])
    Ea_eV = float(np.mean(cfg["sim"]["physics"]["Ea_eV_range"]))
    ratios = [tcfg["split"]["train"], tcfg["split"]["val"], tcfg["split"]["test"]]

    channels = _load_channels(target_h5)
    n_traj = max(c["traj_id"] for c in channels) + 1
    tr_ids, va_ids, te_ids = split_trajectories(n_traj, ratios, seed)

    tr_set, te_set = set(tr_ids), set(te_ids)
    train_ch = [c for c in channels if c["traj_id"] in tr_set]
    test_ch = [c for c in channels if c["traj_id"] in te_set]
    train_failed = [c for c in train_ch if c["event"]]

    # ---- 标定 Arrhenius D_EOL = train 失效通道 EOL 处剂量中位 ----
    D_eol = None
    if train_failed:
        d_eol_list = []
        for c in train_failed:
            Tj_K = c["x_ch"][:, COL_T_DEV] + 273.15
            rate = _arrhenius_rate(Tj_K, Ea_eV)
            D = np.cumsum(rate)
            if c["eol_idx"] < len(D):
                d = float(D[c["eol_idx"]])
                if np.isfinite(d) and d > 0:
                    d_eol_list.append(d)
        if d_eol_list:
            D_eol = float(np.median(d_eol_list))

    # ---- constant: train 失效通道 rul 归一均值 ----
    if train_failed:
        train_rul_norm = np.concatenate([c["rul_ch"] for c in train_failed]) / rul_max
        const_val = float(np.mean(train_rul_norm))
    else:
        const_val = 0.0

    # ---- 相似性库: train 失效通道 p_drift 轨迹 + 对应 rul_remaining ----
    # library_size 限制: 同轨迹 16 子阵互为相关 (来自同一物理退化), 全用既冗余又慢;
    # 按 EOL 分位数均匀采 library_size 条, 覆盖完整寿命分布 (DTW 匹配代表性 + 算力平衡)
    library_full = [(c["x_ch"][:, COL_P_DRIFT], c["rul_ch"], c["eol_idx"])
                    for c in train_failed]
    library_size = int(cfg.get("baselines", {}).get("similarity_library_size", 60))
    if len(library_full) > library_size:
        eols = np.array([t[2] for t in library_full])
        # 按 EOL 分位数采 (覆盖短/中/长寿命)
        qs = np.linspace(0, 1, library_size)
        target_eols = np.quantile(eols, qs)
        # 每个 target EOL 找最近邻
        idx_picked: list[int] = []
        used = set()
        for te in target_eols:
            order = np.argsort(np.abs(eols - te))
            for j in order:
                if int(j) not in used:
                    idx_picked.append(int(j))
                    used.add(int(j))
                    break
        library = [(library_full[j][0], library_full[j][1]) for j in idx_picked[:library_size]]
    else:
        library = [(t[0], t[1]) for t in library_full]

    # ---- 预测 test 通道 ----
    true_norm = np.concatenate([c["rul_ch"] for c in test_ch]) / rul_max
    event_mask = np.concatenate([np.full(len(c["rul_ch"]), c["event"])
                                 for c in test_ch]).astype(bool)
    p_const, p_zext, p_arr, p_sim, p_pf = [], [], [], [], []
    for i, c in enumerate(test_ch):
        rt = c["rul_ch"] / rul_max
        p_const.append(_constant_predict(rt, const_val))
        # z 外推: NaN 用常数兜底 (保所有点可比)
        zext = _z_extrap_predict(c["z_ch"], rul_max, window=50)
        p_zext.append(np.where(np.isfinite(zext), zext, const_val))
        # Arrhenius: 无 D_eol 退化为常数 (无 train 失效场景)
        if D_eol is not None:
            Tj_K = c["x_ch"][:, COL_T_DEV] + 273.15
            arr = _arrhenius_predict(Tj_K, D_eol, rul_max, Ea_eV)
            p_arr.append(np.where(np.isfinite(arr), arr, const_val))
        else:
            p_arr.append(np.full_like(rt, const_val))
        # 相似性匹配
        if library:
            sim = _similarity_predict(c["x_ch"][:, COL_P_DRIFT], library, rul_max,
                                      top_k=5, n_anchors=20, n_grid=32)
            p_sim.append(np.where(np.isfinite(sim), sim, const_val))
        else:
            p_sim.append(np.full_like(rt, const_val))
        # 粒子滤波 (seed 每通道递增, 避免粒子坍缩相关性)
        pf = _particle_filter_predict(c["x_ch"][:, COL_P_DRIFT], rul_max,
                                      n_particles=200, seed=seed + i)
        p_pf.append(pf)
        if verbose and (i + 1) % 25 == 0:
            print(f"    [baseline pred] {i + 1}/{len(test_ch)} 通道", flush=True)

    results = {}
    for name, pred in [("constant", p_const), ("z_extrap", p_zext),
                       ("arrhenius", p_arr), ("similarity_matching", p_sim),
                       ("particle_filter", p_pf)]:
        p = np.concatenate(pred)
        results[name] = _evaluate_split(true_norm, p, event_mask)
    results["_protocol"] = {
        "rul_max_norm": rul_max, "Ea_eV": Ea_eV, "D_eol": D_eol,
        "const_val": const_val,
        "n_train_traj": len(tr_ids), "n_test_traj": len(te_ids),
        "n_train_failed_ch": len(train_failed), "n_test_ch": len(test_ch),
        "n_library": len(library), "seed": seed,
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="checkpoints/baselines_channel.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    res = evaluate_channel_baselines(cfg, seed)
    if res is None:
        return
    proto = res.pop("_protocol")
    print(f"\n===== 通道级基线 (seed={seed}, rul_max_norm={proto['rul_max_norm']:.0f}) =====")
    print(f"协议: Ea={proto['Ea_eV']:.2f}eV  D_EOL={proto['D_eol']}  "
          f"const={proto['const_val']:.4f}")
    print(f"划分: train {proto['n_train_traj']} 轨迹 (失效通道 {proto['n_train_failed_ch']}) "
          f"/ test {proto['n_test_traj']} 轨迹 ({proto['n_test_ch']} 通道); "
          f"similarity library={proto['n_library']}")
    print(f"{'基线':<22} {'RMSE':>8} {'PHM':>10} {'MAE':>8} {'删失违反率':>10}")
    print("-" * 62)
    for name in ["constant", "z_extrap", "arrhenius", "similarity_matching",
                 "particle_filter"]:
        m = res[name]
        print(f"{name:<22} {m['rmse']:>8.4f} {m['phm']:>10.2f} "
              f"{m['mae']:>8.4f} {m['censor_violation_rate']:>10.3f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "_protocol": proto}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n>> {out}")


if __name__ == "__main__":
    main()
