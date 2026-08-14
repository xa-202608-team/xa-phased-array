"""sim/build_array_hi.py

相控阵目标域 HI + 特征 (PA4; 二轮重做修复 GPT 八/九 + 五/六/七)。

latent/telemetry 隔离 (修复 GPT 八 — x 不再含无噪答案变量):
  模型输入只用 telemetry_obs (加噪可观测量); latent (EIRP_norm/IDSS_ratio/damage/dEIRP)
  存于 df 供标签/物理一致性损失, 不进 x。

子阵节点保留 (修复 GPT 九 — 不再把 16 子阵 mean 掉):
  x_global (T, F_global): 阵列级 obs
  x_nodes  (T, 16, F_node): 子阵级 obs 节点 (供 array_aggregator=attention)

HI 固定物理限值 (GPT 六): HI=max(clip(HI_M, HI_SLL, HI_θ)), HI=1 即服务越限
右删失 (GPT 五): event_observed; 未失效 rul 为下界
damage_norm (GPT 七): 不二次归一, event 时 EOL 处=1
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

# 阵列级 obs (模型输入, 全部加噪可观测量 — 不含 EIRP_norm/IDSS 等无噪答案)
X_GLOBAL_COLS = ["G_array_dB", "M_link_dB", "SLL_dB", "theta_err_deg", "Tj", "I_D_obs"]
# 子阵级 obs 节点 (供 attention; sa_feat 列: 0=mean_pow,1=q10,2=I_D,4=amp,5=phase,6=eff)
X_NODE_COLS = ["mean_power", "q10_power", "I_D_node", "amp_rms", "phase_rms", "eff_ratio"]
SA_NODE_IDX = [0, 1, 2, 4, 5, 6]

# 无噪阵列评分真值：仅供后续空间验收读取，绝不进入 x_global/x_nodes 或训练输入。
ARRAY_SCORING_LABELS = {
    "label_array_G_array_dB": ("G_array_dB_true", "dB"),
    "label_array_EIRP_norm": ("EIRP_norm", "ratio"),
    "label_array_SLL_dB": ("SLL_dB_true", "dB"),
    "label_array_theta_err_deg": ("theta_err_deg_true", "deg"),
    "label_array_M_link_dB": ("M_link_dB_true", "dB"),
}
ARRAY_SCORING_LABEL_SCHEMA = "array_scoring_truth_v1"
ARRAY_TWIN_METADATA_SCHEMA = "subarray_array_twin_v1"
TARGET_CONDITION_SCHEMA = "target_ood_conditions_v1"
NODE_STATE_LABELS = {
    "label_node_d_perm": "latent_d_perm",
    "label_node_q_trap": "latent_q_trap",
    "label_node_r_th": "latent_r_th",
}


def build_features(cols: dict, sa: np.ndarray, params: dict,
                   service_limits: dict, cap_ratio: float, rng: np.random.Generator,
                   d_ref_override: float | None = None):
    """返回 (x_global, x_nodes, HI, damage_norm, rul, eol, event_observed)。

    x_global/x_nodes 只用 telemetry_obs (加噪); latent 留在 cols (存 df 供标签/物理损失)。
    """
    n = len(cols["EIRP_norm"])

    # ---- 阵列级 obs (clip 防极端值: 严重退化时 G_array→-300dB 触发数值爆炸) ----
    G_obs = np.clip(np.asarray(cols["G_array_dB"], float), -40.0, 5.0)
    M_obs = np.clip(np.asarray(cols["M_link_dB"], float), -40.0, 5.0)
    SLL_obs = np.clip(np.asarray(cols["SLL_dB"], float), -30.0, 0.0)
    th_obs = np.clip(np.asarray(cols["theta_err_deg"], float), -60.0, 60.0)
    Tj = np.asarray(cols["Tj"], float)
    # I_D_obs: 漏极电流 (可观, 偏置监测) — 从 latent IDSS_ratio 派生 + 测量噪声
    IDSS_latent = np.asarray(cols["IDSS_ratio"], float)
    I_D_obs = IDSS_latent + rng.normal(0, 0.01, n)
    x_global = np.stack([G_obs, M_obs, SLL_obs, th_obs, Tj, I_D_obs], axis=1).astype(np.float32)

    # ---- 子阵级 obs 节点 (加噪, 供 attention) ----
    sa_node = sa[:, :, SA_NODE_IDX].astype(np.float32)            # (T,16,6) latent
    noise_scale = 0.02 * (np.abs(sa_node).mean(axis=(0, 1), keepdims=True) + 1e-6)
    x_nodes = (sa_node + rng.normal(0, 1, sa_node.shape) * noise_scale).astype(np.float32)

    # ---- HI 固定物理限值 (GPT 六) ----
    margin0 = float(params["margin0_dB"])
    M_link_true = np.asarray(cols["M_link_dB_true"], float)
    HI_M = np.clip(1.0 - M_link_true / margin0, 0.0, 1.0)
    SLL_true = np.asarray(cols["SLL_dB_true"], float)
    SLL0 = float(SLL_true[0])
    SLL_max = float(service_limits["SLL_max_dB"])
    HI_SLL = np.clip((SLL_true - SLL0) / (SLL_max - SLL0), 0.0, 1.0)
    theta_true = np.asarray(cols["theta_err_deg_true"], float)
    theta_max = float(service_limits["theta_err_max_deg"])
    HI_theta = np.clip(np.abs(theta_true) / theta_max, 0.0, 1.0)
    HI = np.maximum.reduce([HI_M, HI_SLL, HI_theta]).astype(np.float32)

    # ---- damage_norm + 右删失 (GPT 五/七) ----
    damage = np.asarray(cols["damage"], float)
    lf = np.asarray(cols["label_fail"])
    t = np.arange(n)
    event = bool(lf.any())
    eol = int(np.argmax(lf)) if event else n - 1
    # P0-3 (GPT 审阅): damage_norm 用 main 传入的固定 D_EOL (训练集中位 damage[eol]),
    # 非逐轨迹终点 (旧 d_ref=damage[eol]/[-1] 用未来信息 + 删失强制归一 1)。
    # d_ref_override=None 回退逐轨迹 (兼容旧 smoke/单测)。
    if d_ref_override is not None and d_ref_override > 1e-12:
        d_ref = d_ref_override
    else:
        d_ref = damage[eol] if damage[eol] > 1e-12 else (damage[-1] if damage[-1] > 1e-12 else 1.0)
    damage_norm = np.clip(damage / d_ref, 0.0, 1.0).astype(np.float32)
    if event:
        rul = np.minimum((eol - t).astype(float), cap_ratio * n)
        rul[eol:] = 0.0
    else:
        # P0 (复核第三条): 删失轨迹 rul 同样封顶 cap_ratio*n。旧版 rul=n-1-t 未封顶,
        # 归一后达 2.86 (失效段≤1.0), 模型学会从 t 反推仿真截止时刻 (duration_years=8
        # 人为设定) 而非退化物理。constant RMSE 0.855 反推: 删失段标签均匀 [0,2.86]
        # std 0.83≈0.855 印证。删失 rul 本质是下界, 训练端须配合 hinge (P0-2)
        rul = np.minimum((n - 1 - t).astype(float), cap_ratio * n)
    # P0-1 (GPT 审阅): 失效轨迹截断到 EOL (丢弃 EOL 后 rul=0 段, 根除"失效后窗口"泄漏;
    # 项目要证"失效前提前预测", 不含失效后零标签样本)。保留 [0, eol] 含 EOL 点 (rul=0)。
    if event:
        keep = eol + 1
        x_global, x_nodes, HI = x_global[:keep], x_nodes[:keep], HI[:keep]
        damage_norm, rul = damage_norm[:keep], rul[:keep]
    return x_global, x_nodes, HI, damage_norm, rul.astype(np.float32), eol, event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phased_array.yaml")
    ap.add_argument("--in", dest="indir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cap_ratio = float(cfg["source"]["rul_cap_ratio"])
    lim = cfg["sim"]["service_limits"]
    array_cfg = cfg["sim"]["array"]
    set_seed(cfg["seed"], cfg["reproducibility"]["deterministic"])

    damage_path = cfg.get("sim", {}).get("physics", {}).get("damage_path", "legacy_scalar")
    if args.indir:
        h5_in = ROOT / args.indir / "phased_array_all.h5"
    elif damage_path == "gan_state":
        h5_in = ROOT / cfg["target"]["raw_path"]
    else:
        h5_in = ROOT / "data/simulated/phased_array/sim_v1/seed_42/phased_array_all.h5"
    if not h5_in.exists():
        print(f"!! 缺 {h5_in}; 先 python -m src.sim.phased_array_sim --n_traj 200")
        sys.exit(1)
    if args.out:
        out = ROOT / args.out
    elif damage_path == "gan_state":
        out = ROOT / cfg["target"]["feature_path"]
    else:
        out = ROOT / "data/features/phased_array/schema_v1/target/target_features.h5"
    out.parent.mkdir(parents=True, exist_ok=True)

    needed = ["t", "damage", "EIRP_norm", "G_array_dB", "G_array_dB_true", "M_link_dB", "M_link_dB_true",
              "SLL_dB", "SLL_dB_true", "theta_err_deg", "theta_err_deg_true",
              "IDSS_ratio", "Tj", "duty", "label_fail"]
    rng = np.random.default_rng(cfg["seed"])
    # 与 phased_array_sim 中 label_channel 的 256 元件→16 子阵定义严格一致。
    from src.sim.phased_array_sim import _subarray_ids, _subarray_reduce
    subarray_ids = _subarray_ids(array_cfg["grid"], int(array_cfg["subarray_block"]))
    n_subarrays = int(subarray_ids.max()) + 1
    # P0-3 终修 (GPT 三轮): D_EOL 从 config sim.physics.D_EOL_fixed 读 (仿真器预固定物理阈值,
    # 基于一次全失效中位估计), 不遍历含 test 的 eol (避免 test 泄漏)。缺 cfg 回退遍历 (兼容旧)。
    D_EOL = float(cfg.get("sim", {}).get("physics", {}).get("D_EOL_fixed", 0.0))
    if D_EOL <= 1e-12:   # 回退: 遍历全失效中位 (旧逻辑, 含 test; 仅 cfg 未设时)
        d_eol_list = []
        with h5py.File(h5_in, "r") as fin:
            for key in sorted(fin.keys()):
                lf = fin[key]["label_fail"][:]
                if lf.any():
                    d_eol_list.append(float(fin[key]["damage"][int(np.argmax(lf))]))
        D_EOL = float(np.median(d_eol_list)) if d_eol_list else 1.0
    print(f">> P0-3 D_EOL (cfg 预固定, 不遍历 test) = {D_EOL:.6e}")
    # 第二遍: build with 固定 D_EOL + P0-1 失效截断 (build_features 内)
    dmg_corr, events, n_traj = [], [], 0
    with h5py.File(h5_in, "r") as fin, h5py.File(out, "w") as fout:
        fout.attrs["dynamics_id"] = str(fin.attrs.get("dynamics_id", ""))
        fout.attrs["target_condition_schema"] = TARGET_CONDITION_SCHEMA
        for key in sorted(fin.keys()):
            g = fin[key]
            cols = {c: g[c][:] for c in needed}
            sa = g["subarray_features"][:]
            params = {"margin0_dB": float(g.attrs["margin0_dB"])}
            xg, xn, HI, dmg, rul, eol, ev = build_features(cols, sa, params, lim, cap_ratio, rng, d_ref_override=D_EOL)
            gg = fout.create_group(key)
            keep = len(xg)
            xg_ds = gg.create_dataset("x_global", data=xg)   # (T,6) 阵列级 obs
            xg_ds.attrs["feature_names"] = ",".join(X_GLOBAL_COLS)
            xn_ds = gg.create_dataset("x_nodes", data=xn)    # (T,16,6) 子阵 obs 节点
            xn_ds.attrs["feature_names"] = ",".join(X_NODE_COLS)
            gg.create_dataset("hi_array", data=HI)
            gg.create_dataset("damage_norm", data=dmg)
            gg.create_dataset("rul", data=rul)
            gg.create_dataset("label_fail", data=cols["label_fail"][:keep])
            # 原始物理秒时间轴与 EOL 截断完全同步，供 damage-state transition 的 dt 使用。
            gg.create_dataset("time_s", data=np.asarray(cols["t"], np.float64)[:keep])
            # Gate 2 物理 u_phys 辅助量：逐步 duty_out（随扫描角变化），供源/目标同坐标系
            # 构造 [a_T, s, recovery]；与 time_s 同 EOL 截断，标为 stress 辅助量，绝不进 x。
            duty_ds = gg.create_dataset("physical_duty", data=np.asarray(cols["duty"], np.float32)[:keep])
            duty_ds.attrs["label_level"] = "physical_stress_aux"
            duty_ds.attrs["access"] = "stress_aux_not_model_input"
            # 空间阶段评分真值：与 EOL 截断严格同长度，标为 score-only，禁止作为模型输入。
            for label_name, (raw_name, units) in ARRAY_SCORING_LABELS.items():
                ds = gg.create_dataset(label_name, data=np.asarray(cols[raw_name], np.float32)[:keep])
                ds.attrs["units"] = units
                ds.attrs["label_level"] = "array_scoring_truth"
                ds.attrs["access"] = "score_only_not_model_input"
            # latent (供物理一致性损失/分析, 不进模型输入 x)
            for lc in ["damage", "EIRP_norm", "IDSS_ratio"]:
                gg.create_dataset(f"latent_{lc}", data=np.asarray(cols[lc], np.float32)[:keep])
            # gan_state 原始通道状态和无噪 RF 标签与特征同步截断，绝不进入 x。
            for name in g.keys():
                if name.startswith(("latent_d_perm", "latent_q_trap", "latent_r_th", "label_channel_")):
                    gg.create_dataset(name, data=np.asarray(g[name][:])[:keep])
            # 逐元件 latent 聚合为与 RF label_channel 对齐的子阵状态监督，绝不写入 x。
            for label_name, raw_name in NODE_STATE_LABELS.items():
                if raw_name not in g:
                    continue        # sim_v1 (旧标量路径) 无逐元件 GaN 损伤状态, 跳过
                values = _subarray_reduce(np.asarray(g[raw_name][:keep], np.float32), subarray_ids, n_subarrays)
                ds = gg.create_dataset(label_name, data=values.astype(np.float32))
                ds.attrs["label_level"] = "subarray_train_label"
                ds.attrs["access"] = "train_label_not_model_input"
                ds.attrs["state_name"] = label_name.removeprefix("label_node_")
            gg.attrs["eol_idx"] = eol
            gg.attrs["event_observed"] = int(ev)
            gg.attrs["n_features_global"] = len(X_GLOBAL_COLS)
            gg.attrs["n_features_node"] = len(X_NODE_COLS)
            gg.attrs["array_scoring_label_schema"] = ARRAY_SCORING_LABEL_SCHEMA
            gg.attrs["array_scoring_label_names"] = ",".join(ARRAY_SCORING_LABELS)
            # 确定性阵列孪生所需几何/扫描/链路元数据，显式标为 twin-only，不进 x。
            required_metadata = ("scan_az_deg", "margin0_dB", "duty", "Tj_base_C")
            missing_metadata = [name for name in required_metadata if name not in g.attrs]
            if missing_metadata:
                raise ValueError(f"raw 轨迹 {key} 缺少阵列孪生/OOD 条件元数据: {', '.join(missing_metadata)}")
            gg.attrs["array_twin_metadata_schema"] = ARRAY_TWIN_METADATA_SCHEMA
            gg.attrs["array_twin_metadata_access"] = "twin_only_not_model_input"
            gg.attrs["scan_az_deg"] = float(g.attrs["scan_az_deg"])
            gg.attrs["margin0_dB"] = float(g.attrs["margin0_dB"])
            gg.attrs["duty_cycle"] = float(g.attrs["duty"])
            gg.attrs["Tj_base_C"] = float(g.attrs["Tj_base_C"])
            gg.attrs["array_grid"] = np.asarray(array_cfg["grid"], dtype=np.int32)
            gg.attrs["element_spacing_lambda"] = float(array_cfg["element_spacing_lambda"])
            gg.attrs["subarray_block"] = int(array_cfg["subarray_block"])
            dmg_corr.append(float(np.corrcoef(HI, dmg)[0, 1]))
            events.append(int(ev))
            n_traj += 1

    print(f">> 处理 {n_traj} 条轨迹 -> {out}")
    if args.report:
        dmg_corr = np.array(dmg_corr)
        events_a = np.array(events)
        print("\n===== PA4 阵列 HI 报告 (latent 隔离 + 子阵节点) =====")
        print(f"x_global = (T, {len(X_GLOBAL_COLS)}) 阵列级 obs: {X_GLOBAL_COLS}")
        print(f"x_nodes  = (T, 16, {len(X_NODE_COLS)}) 子阵 obs 节点: {X_NODE_COLS}")
        print(f"latent   = damage/EIRP_norm/IDSS_ratio (不进 x, 供物理损失)")
        print(f"HI-damage 相关 = {dmg_corr.mean():.3f}")
        print(f"event_observed: {events_a.sum()}/{n_traj} 失效, {n_traj-events_a.sum()} 右删失")
        print("HI=max(clip(HI_M,HI_SLL,HI_θ)); x 只用 telemetry_obs (修复 GPT 八/九)")
        print("==========================\n")


if __name__ == "__main__":
    main()
