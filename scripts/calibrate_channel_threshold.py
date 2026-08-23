"""scripts/calibrate_channel_threshold.py

T2b: δ_R 阈值标定 (独立 calibration_seed, 与实验 seed 严格隔离)。

协议 (文档 T2b, 严格遵守防阈值泄漏):
  1. calibration_seed=1234 单独跑仿真 50 轨迹 (与实验 seed 42 数据集完全隔离)
  2. 扫 δ_R ∈ [0.20, 0.50] 步长 0.05, δ_I/δ_g/P_out 固定为器件规范
  3. 选满足三条件的最小 δ_R:
     - 通道失效率 ∈ [0.30, 0.70]
     - median(EOL_ch) < median(EOL_svc) (优雅降级: 器件越限早于服务越限)
     - 失效通道 EOL 分布不集中在头尾 10% (避免退化成全早失效/全删失)
  4. 打印建议值 (脚本只打印, 不自动改 config; 手工写死进 yaml)

不读实验数据集 (sim_v2/seed_42), 只用 calibration_seed 现场仿真。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed                    # noqa: E402
from src.sim.phased_array_sim import sample_params, simulate, _subarray_ids  # noqa: E402
from src.sim.build_channel_hi import build_channel_labels      # noqa: E402


def run_calibration_sim(cfg: dict, n_traj: int, seed: int):
    """独立 seed 跑 subdose 仿真, 返回轨迹列表 [(f_sub, params, eol_svc, failed_svc, T)]。"""
    sim_cfg = dict(cfg["sim"])
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_traj):
        params = sample_params(rng, sim_cfg)
        traj_rng = np.random.default_rng(params["seed_traj"])
        df, _sa, _eol, failed, twin = simulate(params, sim_cfg, traj_rng)
        if twin is None:
            raise RuntimeError("calibration 要求 subarray_dose.enabled=true (config)")
        label_fail = df["label_fail"].values
        eol_svc = int(np.argmax(label_fail)) if label_fail.any() else len(label_fail) - 1
        trajs.append((twin["latent_sub_damage"], params, eol_svc, bool(failed), len(df)))
    return trajs


def evaluate_delta(trajs, deltas, cap_ratio, n_sa):
    """对给定 deltas 算三条件指标 (纯内存, 不写文件)。"""
    n_ch = n_fail = 0
    eol_ch_list, eol_svc_list = [], []
    T_ref = trajs[0][4]
    for f_sub, params, eol_svc, failed_svc, _T in trajs:
        for s in range(n_sa):
            _z, _hi, _rul, event, eol, _keep = build_channel_labels(
                f_sub[:, s], params, deltas, cap_ratio)
            n_ch += 1
            if event:
                n_fail += 1
                eol_ch_list.append(eol)
        if failed_svc:
            eol_svc_list.append(eol_svc)
    fail_rate = n_fail / max(n_ch, 1)
    med_eol_ch = float(np.median(eol_ch_list)) if eol_ch_list else 0.0
    med_eol_svc = float(np.median(eol_svc_list)) if eol_svc_list else 0.0
    if eol_ch_list:
        eol_arr = np.array(eol_ch_list)
        head_frac = float(np.sum(eol_arr < 0.1 * T_ref) / len(eol_arr))
        tail_frac = float(np.sum(eol_arr > 0.9 * T_ref) / len(eol_arr))
        concentrated = max(head_frac, tail_frac)
    else:
        concentrated = 1.0
    return {
        "fail_rate": fail_rate,
        "med_eol_ch": med_eol_ch,
        "med_eol_svc": med_eol_svc,
        "graceful": med_eol_ch < med_eol_svc,
        "concentrated": concentrated,
        "n_fail": n_fail, "n_ch": n_ch,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n_traj", type=int, default=50)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ch = cfg["channel_level"]
    cal_seed = args.seed if args.seed is not None else int(ch["calibration_seed"])
    # F1-A: channel_level.rul_cap_ratio 已从 config 删除 (v2 双字段改不吃 cap_ratio);
    # 该脚本仍调 v1 build_channel_labels (需要 cap_ratio 参数), 故保留 0.35 语义作默认。
    cap_ratio = float(ch.get("rul_cap_ratio", 0.35))
    deltas_base = dict(ch["delta_thresholds"])
    set_seed(cal_seed, cfg["reproducibility"]["deterministic"])

    grid = cfg["sim"]["array"]["grid"]
    block = int(cfg["sim"]["array"]["subarray_block"])
    n_sa = int(_subarray_ids(grid, block).max()) + 1

    print(f">> T2b δ_R 标定 (calibration_seed={cal_seed}, n_traj={args.n_traj}, 独立于实验 seed)")
    print(f">> 扫描 δ_R ∈ [0.20, 0.50] 步长 0.05; δ_I={deltas_base['I_DSS']}, "
          f"δ_g={deltas_base['g_m']}, δ_P={deltas_base['P_out']} 固定")
    trajs = run_calibration_sim(cfg, args.n_traj, cal_seed)
    n_fail_svc = sum(1 for t in trajs if t[3])
    print(f">> 标定数据集: {len(trajs)} 轨迹, 服务失效 {n_fail_svc}/{len(trajs)}")

    hdr = f"{'δ_R':>5} {'失效率':>8} {'med_EOL_ch':>11} {'med_EOL_svc':>12} {'优雅降级':>9} {'头尾集中':>9} {'满足':>5}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    candidates = []
    for dR in np.arange(0.20, 0.5001, 0.05):
        deltas = dict(deltas_base)
        deltas["R_DS"] = round(float(dR), 3)
        m = evaluate_delta(trajs, deltas, cap_ratio, n_sa)
        ok = (0.30 <= m["fail_rate"] <= 0.70
              and m["graceful"]
              and m["concentrated"] < 0.30)
        print(f"{dR:>5.2f} {m['fail_rate']:>8.1%} {m['med_eol_ch']:>11.0f} {m['med_eol_svc']:>12.0f} "
              f"{'是' if m['graceful'] else '否':>9} {m['concentrated']:>9.1%} {'✓' if ok else '':>5}")
        if ok:
            candidates.append((deltas["R_DS"], m))

    print("\n===== T2b 标定结论 =====")
    if candidates:
        best_dR, best_m = candidates[0]          # 满足条件的最小 δ_R
        print(f">> 建议 δ_R = {best_dR:.2f} (满足三条件的最小值)")
        print(f">>   失效率={best_m['fail_rate']:.1%}, EOL_ch={best_m['med_eol_ch']:.0f} < "
              f"EOL_svc={best_m['med_eol_svc']:.0f}, 头尾集中={best_m['concentrated']:.1%}")
        print(f">> 手工写死进 configs/phased_array.yaml → channel_level.delta_thresholds.R_DS")
        print(f">> 然后在实验数据集 (seed 42) 上重算三条件验证 (build_channel_hi --report)")
    else:
        print(">> 无 δ_R 同时满足三条件 (失效率∈[0.30,0.70] ∧ EOL_ch<EOL_svc ∧ 头尾<30%)")
        print(">> 回退 (文档 §8): 放宽失效率到 [0.25,0.75], 或调 sigma_sub↓ / Delta_R↑ / dropout_thr↑")
    print(f"==========================\n")


if __name__ == "__main__":
    main()
