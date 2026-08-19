"""sim/build_channel_hi.py

通道级 HI + canonical 特征构建 (T2/M2, channel_level 迁移路线主线)。

与 build_array_hi.py 的区别:
  - 层级: 子阵/通道级 (器件层), 非阵列服务级 — 迁移接口下沉到器件层
  - 标签源: latent_sub_damage f_s (M1 子阵级损伤真值) → z=max(dR/δR,dI/δI,dg/δg)
  - 特征: canonical device schema 4 维 (p_drift_norm/T_dev_C/duty/drive_norm)
    第0维 = 归一化到各自器件失效阈值的关键参量漂移 (源/目标同语义, 迁移核心)
  - 输入: sim_v2 (subdose 数据集, DYNAMICS_ID=phased_array_subdose_v2)
  - 输出: schema_ch_v1/target/channel_features.h5, 每轨迹每子阵一组

不改 build_array_hi.py (cross_level_transfer 复跑用, 保留)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils import load_config, set_seed            # noqa: E402

CANONICAL_COLS = ["p_drift_norm", "T_dev_C", "duty", "drive_norm"]
CANONICAL_SCHEMA = "device_canonical_v1"
CHANNEL_LABEL_SCHEMA = "channel_label_v1"
# sa_feat 列 (与 phased_array_sim _simulate_subdose / simulate 一致):
# 0=mean_pow, 1=q10_pow, 2=IDSS, 3=Tj, 4=amp_rms, 5=phase_rms, 6=eff_ratio, 7=q90_f
SA_COL_POWER = 0
SA_COL_IDSS = 2
SA_COL_TJ = 3
SA_COL_AMP = 4
# 需要从轨迹 attrs 读取的物理参数 (M1 main 已写入)
PARAM_ATTRS = ["Delta_R", "decay_I", "decay_g", "dphi_max_deg", "Ea_eV",
               "Tj_base_C", "Tj_amp_C", "life_scale_years", "weibull_beta",
               "scan_az_deg", "margin0_dB", "duty"]
TWIN_KEYS = ["twin_c_elem", "twin_eta_R", "twin_eta_phi",
             "twin_dropout_thr", "twin_grad_dir", "twin_subarray_ids"]


def build_channel_labels(f_sub_s: np.ndarray, params: dict, deltas: dict,
                         cap_ratio: float):
    """由子阵损伤真值 f_s (T,) 派生通道级 z/hi/rul/event/eol/keep。

    z = max(dR/δR, dI/δI, dg/δg);  hi=clip(z,0,1);  rul 封顶 cap_ratio*T。
    失效通道截断到 EOL (P0-1 铁律: 丢弃 EOL 后零标签窗, 根除泄漏)。
    """
    Delta_R = float(params["Delta_R"])
    decay_I = float(params["decay_I"])
    decay_g = float(params["decay_g"])
    dR = (Delta_R - 1.0) * f_sub_s
    dI = decay_I * f_sub_s
    dg = decay_g * f_sub_s
    z = np.maximum.reduce([dR / deltas["R_DS"], dI / deltas["I_DSS"], dg / deltas["g_m"]])
    T = len(z)
    hi = np.clip(z, 0.0, 1.0)
    event = bool((z >= 1.0).any())
    t = np.arange(T)
    if event:
        eol = int(np.argmax(z >= 1.0))
        rul = np.minimum((eol - t).astype(float), cap_ratio * T)
        rul[eol:] = 0.0
        keep = eol + 1
    else:
        eol = T - 1
        rul = np.minimum((T - 1 - t).astype(float), cap_ratio * T)
        keep = T
    return z.astype(np.float32), hi.astype(np.float32), rul[:keep].astype(np.float32), \
        event, eol, keep


def build_canonical_x(sa_feat_s: np.ndarray, duty: float, deltas: dict):
    """子阵 sa_feat (T,8) → canonical 4 维 x_ch (T,4)。

    第0维 p_drift_norm = max(ΔIDSS/δI, ΔP/δP): 归一化到器件失效阈值的参量漂移,
    与源域 (ΔR_DS/δR_src) 同语义 — 迁移成立的核心可观测量。
    """
    T = sa_feat_s.shape[0]
    IDSS = sa_feat_s[:, SA_COL_IDSS]
    P = sa_feat_s[:, SA_COL_POWER]
    IDSS0 = IDSS[0] if abs(IDSS[0]) > 1e-9 else 1.0
    P0 = P[0] if abs(P[0]) > 1e-9 else 1.0
    dIDSS = np.clip(1.0 - IDSS / IDSS0, 0.0, None)
    dP = np.clip(1.0 - P / P0, 0.0, None)
    p_drift = np.maximum(dIDSS / deltas["I_DSS"], dP / deltas["P_out"])
    T_dev = sa_feat_s[:, SA_COL_TJ] - 273.15  # 修复: 仿真器输出 Kelvin 转为 °C，与命名 T_dev_C 一致
    amp = sa_feat_s[:, SA_COL_AMP]
    amp0 = amp[0] if abs(amp[0]) > 1e-9 else 1.0
    drive = amp / amp0
    x = np.stack([p_drift, T_dev, np.full(T, duty), drive], axis=1).astype(np.float32)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--in", dest="indir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ch = cfg["channel_level"]
    deltas = dict(ch["delta_thresholds"])
    cap_ratio = float(ch["rul_cap_ratio"])
    set_seed(cfg["seed"], cfg["reproducibility"]["deterministic"])

    seed = cfg["seed"]
    indir = ROOT / (args.indir or f"data/simulated/phased_array/sim_v2/seed_{seed}")
    h5_in = indir / "phased_array_all.h5"
    if not h5_in.exists():
        print(f"!! 缺 {h5_in}; 先 python -m src.sim.phased_array_sim --n_traj 200")
        sys.exit(1)
    out = ROOT / (args.out or ch["feature_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    from src.sim.phased_array_sim import _subarray_ids
    grid = cfg["sim"]["array"]["grid"]
    block = int(cfg["sim"]["array"]["subarray_block"])
    n_sa = int(_subarray_ids(grid, block).max()) + 1

    stats = {"n_traj": 0, "n_ch": 0, "n_fail": 0, "eol_ch": [], "eol_svc": []}
    with h5py.File(h5_in, "r") as fin, h5py.File(out, "w") as fout:
        fout.attrs["dynamics_id"] = str(fin.attrs.get("dynamics_id", ""))
        fout.attrs["canonical_schema"] = CANONICAL_SCHEMA
        fout.attrs["t_dev_unit"] = "degC"      # canonical 第1维 T_dev_C 单位 = sim Tj (Kelvin) − 273.15
        fout.attrs["t_dev_conversion"] = "subarray_features.Tj_K_minus_273.15"
        fout.attrs["channel_label_schema"] = CHANNEL_LABEL_SCHEMA
        fout.attrs["delta_thresholds"] = ",".join(f"{k}={v}" for k, v in deltas.items())
        for key in sorted(fin.keys()):
            g = fin[key]
            sa_feat = g["subarray_features"][:]              # (T, 16, 8) 子阵遥测
            if "latent_sub_damage" in g:
                f_sub = g["latent_sub_damage"][:]           # (T, 16) 子阵损伤真值 (M1)
            else:
                # legacy 标量动力学 (B 路线 §4e, sim_v1): 16 子阵共享阵列级损伤标量
                # 广播 — 不是伪造, 是 legacy 物理的忠实表达 (损伤自由度 1 vs subdose 16,
                # 即 B 路线受控域差距本体); 子阵差异仍来自 subarray_features 观测散布
                f_sub = np.repeat(g["damage"][:][:, None], sa_feat.shape[1], axis=1)
            label_fail = g["label_fail"][:]                  # (T,) 服务越限标记
            params = {k: float(g.attrs[k]) for k in PARAM_ATTRS if k in g.attrs}
            eol_svc = int(np.argmax(label_fail)) if label_fail.any() else sa_feat.shape[0] - 1
            failed_svc = bool(label_fail.any())

            traj_grp = fout.create_group(key)
            for a in PARAM_ATTRS:
                if a in g.attrs:
                    traj_grp.attrs[a] = float(g.attrs[a])
            traj_grp.attrs["array_grid"] = np.asarray(grid, dtype=np.int32)
            traj_grp.attrs["element_spacing_lambda"] = float(cfg["sim"]["array"]["element_spacing_lambda"])
            traj_grp.attrs["subarray_block"] = block
            traj_grp.attrs["eol_svc"] = eol_svc              # 服务 EOL (供 EOL_ch<EOL_svc 优雅降级验证)
            traj_grp.attrs["failed_svc"] = int(failed_svc)
            # 透传 twin 静态量 (twin_only, 不进模型输入; 供 M4 物理孪生前推)
            twin_grp = traj_grp.create_group("twin")
            for tk in TWIN_KEYS:
                if tk in g:
                    twin_grp.create_dataset(tk, data=g[tk][:])

            traj_id = int(key.split("_")[1])
            for s in range(n_sa):
                f_s = f_sub[:, s]
                z, hi, rul, event, eol, keep = build_channel_labels(
                    f_s, params, deltas, cap_ratio)
                x_ch = build_canonical_x(sa_feat[:keep, s, :], params["duty"], deltas)
                sub_grp = traj_grp.create_group(f"sub_{s:02d}")
                sub_grp.create_dataset("x_ch", data=x_ch)
                sub_grp.create_dataset("hi_ch", data=hi[:keep])
                sub_grp.create_dataset("z_ch", data=z[:keep])
                sub_grp.create_dataset("rul_ch", data=rul)
                sub_grp.attrs["event_observed"] = int(event)
                sub_grp.attrs["eol_idx"] = eol
                sub_grp.attrs["traj_id"] = traj_id
                sub_grp.attrs["sub_id"] = s
                sub_grp.attrs["feature_names"] = ",".join(CANONICAL_COLS)
                sub_grp.attrs["canonical_schema"] = CANONICAL_SCHEMA
                stats["n_ch"] += 1
                if event:
                    stats["n_fail"] += 1
                    stats["eol_ch"].append(eol)
            if failed_svc:
                stats["eol_svc"].append(eol_svc)
            stats["n_traj"] += 1

    print(f">> 处理 {stats['n_traj']} 轨迹 × {n_sa} 子阵 = {stats['n_ch']} 通道 -> {out}")
    if args.report:
        fail_rate = stats["n_fail"] / max(stats["n_ch"], 1)
        print("\n===== M2 通道级 HI 报告 =====")
        print(f"canonical schema = {CANONICAL_SCHEMA}, 4 维: {CANONICAL_COLS}")
        print(f"δ_thresholds: {deltas}")
        print(f"通道失效率 = {stats['n_fail']}/{stats['n_ch']} ({100*fail_rate:.1f}%)")
        if stats["eol_ch"] and stats["eol_svc"]:
            med_ch = float(np.median(stats["eol_ch"]))
            med_svc = float(np.median(stats["eol_svc"]))
            print(f"median EOL_ch = {med_ch:.0f} vs median EOL_svc = {med_svc:.0f} "
                  f"(优雅降级 EOL_ch<EOL_svc: {'是' if med_ch < med_svc else '否'})")
        print("z=max(dR/δR,dI/δI,dg/δg); hi=clip(z,0,1); 第0维 p_drift_norm 与源域同语义")
        print("==========================\n")


if __name__ == "__main__":
    main()
