"""baselines/phased_array_baselines.py

相控阵三个非学习基线 (P0 实验 C, plan §"接下来 72 小时")。

判决性用途: 若常数预测 RMSE 与深度模型 (target_only_tcn ~0.47, rul_max_norm 口径)
同量级, 则四组模型没在学退化动力学, 只在记 EOL 后的 0 尾部 (目标域 rul==0 占 42.5%);
后续所有迁移增益对比 (±0.005 量级) 都无意义, 必须先重建目标域数据 (遥测分层/切断答案变量)。

三个基线:
  1. constant   ŷ ≡ mean(rul_train)            愚蠢基线, 模型下界
  2. hi_extrap  hi_array 滑窗线性外推到 HI=1.0   物理启发但非学习
  3. arrhenius  Tj Arrhenius 剂量积分外推        纯物理机理基线

评估协议与 run_groups 完全一致 (轨迹级 split + rul_max_norm 归一 + RMSE/PHM/MAE),
保证基线数字可直接与模型表对照。

用法:
  python -m src.baselines.phased_array_baselines --config configs/phased_array.yaml
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
from src.utils import load_config, set_seed                                   # noqa: E402
from src.baselines.physical_extrap import rmse, mae, phm_score                # noqa: E402
from src.transfer.train_transfer import split_trajectories                   # noqa: E402

K_BOLTZMANN_eVperK = 8.617333262e-5


# ---------------------------------------------------------------- 基线 1: 常数
def _constant_predict(rul_test_norm: np.ndarray, const_val: float) -> np.ndarray:
    """所有 test 点预测同一常数 (train 集 rul 归一均值)。"""
    return np.full_like(rul_test_norm, const_val, dtype=float)


# ---------------------------------------------------------------- 基线 2: HI 外推
def _hi_extrap_predict(hi: np.ndarray, rul_max: float, window: int = 50) -> np.ndarray:
    """hi_array 滑窗线性外推到 HI=1.0 (服务越限阈值), 估计剩余窗数。

    复用 physical_extrap_rul 思路: 对每个 t, 用 [t-window, t] 的 hi 线性拟合,
    外推失效时刻 t_fail=(1.0-intercept)/slope, RUL=t_fail-t。单位=窗 (与 rul 字段一致)。
    早期点 (<window) 或 slope<=0 时返回 NaN (评估时用常数兜底, 保所有点可比)。
    """
    n = len(hi)
    rul = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window)
        seg = hi[lo:i + 1]
        if len(seg) < 5:
            continue
        slope, intercept = np.polyfit(np.arange(len(seg)), seg, 1)
        if slope <= 1e-9:                 # hi 不上升, 无法外推 (健康早期)
            continue
        t_fail = (1.0 - intercept) / slope + lo
        rul[i] = max(0.0, t_fail - i)
    return rul / max(rul_max, 1.0)        # 归一到 rul_max_norm 空间


# ---------------------------------------------------------------- 基线 3: Arrhenius 剂量
def _cum_dose(Tj_K: np.ndarray, Ea_eV: float) -> np.ndarray:
    """Arrhenius 累积剂量 (相对量, dt=1 窗): D(t)=Σ exp(-Ea/(k_B·Tj))。

    纯相对积分; 绝对尺度由 D_EOL 标定消去。Tj 单位开尔文。
    """
    rate = np.exp(-Ea_eV / (K_BOLTZMANN_eVperK * Tj_K))
    return np.cumsum(rate)


def _arrhenius_predict(Tj_K: np.ndarray, D_eol: float, rul_max: float,
                       Ea_eV: float) -> np.ndarray:
    """Arrhenius 剂量积分外推: RUL=(D_EOL-D_now)/rate_now (剩余剂量/当前剂量率)。

    单位=窗数 (rate=每窗剂量, D 标定到 train 失效轨迹 EOL)。早期 rate 噪声大,
    D_now>=D_EOL (已超 train 标定失效) 时 clip 0。
    """
    rate = np.exp(-Ea_eV / (K_BOLTZMANN_eVperK * Tj_K))
    D = np.cumsum(rate)
    remaining = (D_eol - D) / np.maximum(rate, 1e-12)
    return np.clip(remaining, 0.0, None) / max(rul_max, 1.0)


# ---------------------------------------------------------------- 评估
def evaluate_phased_array_baselines(cfg: dict, seed: int) -> dict:
    """跑三个非学习基线, 返回 {constant, hi_extrap, arrhenius} 各自 RMSE/PHM/MAE。

    轨迹级 split 与 run_groups 一致 (train/val/test = config transfer.split);
    rul 按 transfer.rul_max_norm 归一; 评估仅 test 集。
    """
    set_seed(seed, cfg["reproducibility"]["deterministic"])
    tcfg = cfg["transfer"]
    target_h5 = ROOT / tcfg.get(
        "target_feature_path",
        "data/features/phased_array/schema_v1/target/target_features.h5")
    if not target_h5.exists():
        print(f"!! 缺 {target_h5}; 先 python -m src.sim.build_array_hi --report")
        return None
    rul_max = float(tcfg.get("rul_max_norm", 1.0))
    Ea_eV = float(np.mean(cfg["sim"]["physics"]["Ea_eV_range"]))   # GaN HEMT 激活能中值
    ratios = [tcfg["split"]["train"], tcfg["split"]["val"], tcfg["split"]["test"]]

    # ---- 读全部轨迹 (x_global 取 Tj=col4; hi_array; rul; attrs eol/event) ----
    trajs = []
    with h5py.File(target_h5, "r") as f:
        for k in sorted(f.keys()):
            g = f[k]
            trajs.append(dict(
                xg=g["x_global"][:].astype(float),
                hi=g["hi_array"][:].astype(float),
                rul=g["rul"][:].astype(float),
                eol=int(g.attrs.get("eol_idx", 0)),
                event=int(g.attrs.get("event_observed", 0)),
            ))
    n_traj = len(trajs)
    tr_ids, va_ids, te_ids = split_trajectories(n_traj, ratios, seed)

    # ---- 标定 Arrhenius D_EOL = train 失效轨迹 EOL 处剂量的中位 ----
    train_failed = [trajs[i] for i in tr_ids if trajs[i]["event"] == 1]
    D_eol = None
    if train_failed:
        d_eol_list = [_cum_dose(t["xg"][:, 4], Ea_eV)[t["eol"]] for t in train_failed]
        d_eol_list = [d for d in d_eol_list if np.isfinite(d) and d > 0]
        if d_eol_list:
            D_eol = float(np.median(d_eol_list))

    # ---- 常数: train 集 rul 归一均值 ----
    train_rul_norm = np.concatenate([trajs[i]["rul"] for i in tr_ids]) / rul_max
    const_val = float(np.mean(train_rul_norm))

    # ---- test 集预测 (P0-2: 仅失效轨迹; 删失无精确 RUL 不评估, 与 run_groups eval_test 同口径) ----
    test_true, p_const, p_hi, p_arr = [], [], [], []
    for i in te_ids:
        t = trajs[i]
        if t["event"] != 1:            # 删失轨迹跳过 (无精确 RUL, 和模型 eval_test 一致)
            continue
        rt = t["rul"] / rul_max
        test_true.append(rt)
        p_const.append(_constant_predict(rt, const_val))
        hi_p = _hi_extrap_predict(t["hi"], rul_max, window=50)
        hi_p = np.where(np.isfinite(hi_p), hi_p, const_val)
        p_hi.append(hi_p)
        if D_eol is not None:
            arr_p = _arrhenius_predict(t["xg"][:, 4], D_eol, rul_max, Ea_eV)
            arr_p = np.where(np.isfinite(arr_p), arr_p, const_val)
            p_arr.append(arr_p)
        else:
            p_arr.append(np.full_like(rt, const_val))   # 无 train 失效 → 退化为常数

    true = np.concatenate(test_true)
    results = {}
    for name, pred in [("constant", p_const), ("hi_extrap", p_hi), ("arrhenius", p_arr)]:
        p = np.concatenate(pred)
        results[name] = {"rmse": rmse(p, true), "phm": phm_score(p, true),
                         "mae": mae(p, true), "n_traj_test": len(te_ids)}
    # const=0: 最强纯常数判决基线 (test rul==0 占 ~47%, "永远预测失效"的 RMSE 下界)。
    # 若模型 RMSE 贴近 const=0, 说明模型只在记 EOL 后 0 尾部; 远低于则证明学到退化动力学。
    p_zero = np.zeros_like(true)
    results["constant_zero"] = {"rmse": rmse(p_zero, true), "phm": phm_score(p_zero, true),
                                "mae": mae(p_zero, true), "n_traj_test": len(te_ids)}
    results["_protocol"] = {"rul_max_norm": rul_max, "Ea_eV": Ea_eV,
                            "D_eol": D_eol, "const_val": const_val,
                            "n_train": len(tr_ids), "n_test": len(te_ids),
                            "train_failed": len(train_failed)}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="checkpoints/baselines_phased_array.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    res = evaluate_phased_array_baselines(cfg, seed)
    if res is None:
        return
    # 参照: target_only_tcn RMSE (rul_max_norm 口径, 来自 docs/results_phased_array.md)
    ref = 0.4732
    print(f"\n===== 相控阵非学习基线 (seed={seed}, rul_max_norm 口径) =====")
    proto = res.pop("_protocol")
    print(f"协议: rul_max={proto['rul_max_norm']:.0f} Ea={proto['Ea_eV']:.2f}eV "
          f"D_EOL={proto['D_eol']} const={proto['const_val']:.4f}")
    print(f"划分: train {proto['n_train']} (失效 {proto['train_failed']}) / test {proto['n_test']}")
    print(f"{'基线':<14} {'RMSE':>8} {'PHM':>10} {'MAE':>8}")
    print("-" * 44)
    for name in ["constant_zero", "constant", "hi_extrap", "arrhenius"]:
        m = res[name]
        flag = ""
        if name == "constant_zero":
            flag = "  <<< 最强纯常数 (记 0 下界)"
        elif name == "constant":
            flag = "  <<< train 均值常数"
        print(f"{name:<14} {m['rmse']:>8.4f} {m['phm']:>10.2f} {m['mae']:>8.4f}{flag}")
    best_base = min(res[n]["rmse"] for n in
                    ["constant_zero", "constant", "hi_extrap", "arrhenius"])
    print("-" * 42)
    print(f"参照 target_only_tcn RMSE = {ref:.4f} (rul_max_norm 口径, 5 seed mean)")
    print(f"最强非学习基线 RMSE = {best_base:.4f}")
    if best_base < ref:
        print(f"  [!] 非学习基线 优于 深度模型 ({best_base:.4f} < {ref:.4f})")
        print(f"    → 四组模型没在学退化动力学, 仅记 EOL 后 0 尾部 (rul==0 占 42.5%)")
        print(f"    → 后续迁移增益对比 (±0.005) 无意义, 先重建目标域数据")
    else:
        gap = ref / max(best_base, 1e-9)
        print(f"  模型 vs 最强基线 = {ref:.4f}/{best_base:.4f} = {gap:.2f}x (模型有增量学习)")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "_protocol": proto, "_ref_target_only_tcn": ref},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n>> {out}")


if __name__ == "__main__":
    main()
