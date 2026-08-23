# -*- coding: utf-8 -*-
"""从 configs/phased_array.yaml 与 src/sim 代码事实导出三级物理链冻结文档。

用法：python scripts/generate_physics_chain.py [--config configs/phased_array.yaml]
输出：docs/simulation/PHYSICS_CHAIN.yaml（内容与 config 实际加载值联动，不手抄）。

三级定义（与 src/sim/phased_array_sim.py 实现一致）：
  gan_tr: 电参数/温度/辐照/退化状态 -> 通道增益和相位（Arrhenius+Coffin-Manson 损伤积分）
  array: 子阵通道状态 -> 阵列因子、增益、旁瓣、优雅降级（完整 AF 方位切面）
  link:  阵列输出 -> 链路余量、服务状态、失效真值（M_link/SLL/θ 多维越限）
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_chain(cfg: dict, config_path: Path) -> dict:
    sim = cfg["sim"]
    physics = sim["physics"]
    array_cfg = sim["array"]
    sample_h = float(sim["sample_period_s"]) / 3600.0
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()

    chain = {
        "schema": "phased-array-physics-chain/1.0",
        "seed": int(cfg["seed"]),
        "dynamics_id": {
            "subdose_on": "phased_array_subdose_v2",
            "subdose_off": "leo_coupled_v1",
            "source": "src/sim/phased_array_sim.py::DYNAMICS_ID",
        },
        "provenance": {
            "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "config_sha256": config_sha,
            "generator": "scripts/generate_physics_chain.py",
            "note": "结构与取值由本脚本从 config 加载值导出；改动 config 后需重新生成并人工核对",
        },
        "levels": {
            "gan_tr": {
                "title": "GaN T/R 器件级：电参数/温度/辐照/退化状态 -> 通道增益和相位",
                "inputs": [
                    "Ea_eV (激活能)", "Tj_base_C/Tj_amp_C/Tj_drift_C_per_year (结温轨道热循环)",
                    "duty (PA 占空比)", "life_scale_years (器件寿命预算)",
                    "hotspot_field (二维热点温升场)", "damage_model.coffin_manson (热循环疲劳)",
                    "damage_model.rth_feedback (R_th 正反馈)", "subarray_dose (子阵独立剂量)",
                ],
                "outputs": [
                    "f_sub (T,16) 子阵累积损伤真值", "f_ch (T,256) 元件级损伤",
                    "RDS_ratio/IDSS_ratio/gm_ratio (器件参数漂移)",
                    "P_ratio + dropout -> 通道幅度 a、相位 dphi_rad",
                ],
                "truth_fields": [
                    "latent_sub_damage (H5 数据集, 子阵损伤真值)",
                    "damage (df 列, 轨迹级标量真值)",
                    "RDS_drift / IDSS_ratio / gm_ratio (df 列, 无噪真值)",
                    "twin_* 静态孪生量 (twin_only, 不进模型输入)",
                ],
                "observable_fields": [
                    "Tj / Tj_max / Tj_min (df 列)", "duty (df 列)",
                    "RDS_drift / IDSS_ratio / gm_ratio (df 列, 遥测口径)",
                ],
                "prediction_time_observable": ["Tj", "duty"],
                "units": {
                    "damage": "无量纲 (eff_age / life_scale)",
                    "Ea_eV": "eV",
                    "Tj": "degC",
                    "duty": "无量纲 (0.3-0.8)",
                    "RDS_drift": "无量纲 (x 初始值)",
                    "IDSS_ratio / gm_ratio": "无量纲 (比例)",
                    f"遥测采样": f"{int(sim['sample_period_s'])} s = {sample_h:g} h/窗",
                    f"物理积分步": f"{int(sim.get('physics_dt_s', 300))} s",
                },
                "config_keys": [
                    "sim.physics.Ea_eV_range", "sim.physics.time_exp_m",
                    "sim.physics.Tj_range_C", "sim.physics.duty_cycle_range",
                    "sim.physics.damage_model.coffin_manson.enabled",
                    "sim.physics.damage_model.coffin_manson.deltaT_ref_K",
                    "sim.physics.damage_model.coffin_manson.exponent_n",
                    "sim.physics.damage_model.rth_feedback.enabled",
                    "sim.physics.damage_model.rth_feedback.a5",
                    "sim.physics.damage_model.rth_feedback.Trise_per_Rth_K",
                    "sim.physics.hotspot_field.enabled",
                    "sim.physics.subarray_dose.enabled",
                    "sim.physics.subarray_dose.sigma_sub",
                    "sim.physics.life_scale_sigma",
                    "sim.physics.device_spread_sigma",
                    "sim.physics.D_EOL_fixed",
                    "sim.physics.channel_dropout_weibull.enabled",
                    "sim.duration_years", "sim.sample_period_s", "sim.physics_dt_s",
                ],
                "random_sources": [
                    "np.random.default_rng(seed): sample_params 轨迹参数 (Ea/Tj/duty/Delta_R/scan_az/margin0)",
                    "params.seed_traj 独立子流: life_scale 对数正态 / hotspot 场 / "
                    "eta_phi / eta_R / grad_dir / Weibull 退出阈值 / 子阵 sigma_sub 散布",
                ],
                "generation_functions": [
                    "src/sim/phased_array_sim.py::sample_params",
                    "src/sim/phased_array_sim.py::simulate (第一级, 标量路径 subdose off)",
                    "src/sim/phased_array_sim.py::_simulate_subdose (第一级, 子阵独立剂量 subdose on)",
                ],
            },
            "array": {
                "title": "阵列级：子阵通道状态 -> 阵列因子、增益、旁瓣、优雅降级",
                "inputs": [
                    "通道幅度 a (T,256) 与相位 dphi_rad (T,256)",
                    "阵列几何 grid/element_spacing_lambda (16x16, 0.5 波长)",
                    "scan_az_deg 扫描角", "dropout_mask 通道退出 (优雅降级)",
                ],
                "outputs": [
                    "阵列因子方向图 (方位切面 THETA_COARSE_DEG)",
                    "G_array_dB (阵列增益)", "SLL_dB (第一旁瓣, 主瓣窗=左右首零点)",
                    "theta_err_deg (波束指向误差)", "k_failed (失效通道数)",
                    "subarray_features (T,16,8) 子阵遥测特征",
                ],
                "truth_fields": [
                    "G_array_dB_true / SLL_dB_true / theta_err_deg_true (df 列, 无噪)",
                    "EIRP_norm (df 列, 无噪)", "k_failed (df 列, 由 dropout 推出)",
                    "label_channel_gain_dB / phase_deg / Pout_dBm / PAE (gan_state 路径, 子阵 RF 标签)",
                ],
                "observable_fields": [
                    "G_array_dB / SLL_dB / theta_err_deg (df 列, 加噪观测)",
                    "subarray_features 8 维: mean_pow/q10_pow/IDSS/Tj/amp_rms/phase_rms/eff_ratio/q90_f",
                ],
                "prediction_time_observable": ["G_array_dB", "SLL_dB", "theta_err_deg"],
                "units": {
                    "G_array_dB": "dB (相对健康阵峰值归一)",
                    "SLL_dB": "dB",
                    "theta_err_deg": "deg",
                    "k_failed": "通道数 (0-256)",
                    "element_spacing_lambda": f"波长 x {array_cfg['element_spacing_lambda']}",
                    "phase": "deg (存储/输出) 与 rad (计算)",
                },
                "config_keys": [
                    "sim.array.n_elements", "sim.array.grid",
                    "sim.array.element_spacing_lambda", "sim.array.scan_az_deg_range",
                    "sim.array.scan_el_deg", "sim.array.subarray_block",
                    "sim.array_factor.mode",
                    "sim.physics.R_DS_drift_Delta_range", "sim.physics.IDSS_decay_range",
                    "sim.physics.gm_decay_range", "sim.physics.phase_drift_deg_range",
                    "sim.physics.channel_dropout_weibull.beta",
                    "sim.disturbance.spatial_cluster_prob",
                ],
                "random_sources": [
                    "继承 gan_tr 级 traj_rng 子流 (eta_phi / eta_R / grad_dir / Weibull 阈值 / 空间聚簇判定)",
                    "无新增独立随机源 (方向图计算确定性)",
                ],
                "generation_functions": [
                    "src/sim/phased_array_sim.py::_grid_positions",
                    "src/sim/phased_array_sim.py::_subarray_ids",
                    "src/sim/phased_array_sim.py::_array_pattern",
                    "src/sim/phased_array_sim.py::_refine_peak",
                ],
            },
            "link": {
                "title": "链路级：阵列输出 -> 链路余量、服务状态、失效真值",
                "inputs": [
                    "G_array_true 序列 (阵列级输出)", "margin0_dB (初始链路余量)",
                    "service_limits (SLL_max / theta_err_max / consecutive_windows)",
                ],
                "outputs": [
                    "dEIRP_true = G_array_true[0] - G_array_true",
                    "M_link_true = margin0_dB - dEIRP_true (链路余量)",
                    "服务失效判定: 连续 N 窗 M_link<=0 或 SLL>SLL_max 或 |theta_err|>theta_max",
                    "label_fail (0/1) / eol_idx / failed (H5 attrs)",
                ],
                "truth_fields": [
                    "M_link_dB_true (df 列, 无噪)",
                    "label_fail (df 列, 服务失效真值标记)",
                    "eol_idx / failed (H5 group attrs)",
                ],
                "observable_fields": [
                    "M_link_dB (df 列, 加噪观测, 相控阵单指标接入主遥测为 G_array_dB 时 M_link 作链路侧遥测)",
                ],
                "prediction_time_observable": ["M_link_dB"],
                "units": {
                    "M_link_dB": "dB",
                    "label_fail": "0/1",
                    "SLL_max_dB": "dB",
                    "theta_err_max_deg": "deg",
                    "consecutive_windows": f"窗 (x {sample_h:g} h)",
                },
                "config_keys": [
                    "sim.link_budget.initial_margin_dB_range",
                    "sim.service_limits.SLL_max_dB",
                    "sim.service_limits.theta_err_max_deg",
                    "sim.service_limits.consecutive_windows",
                    "sim.disturbance.noise_ratio",
                ],
                "random_sources": [
                    "traj_rng 观测噪声: G_array/SLL/M_link 加 noise_ratio 满量程噪声, "
                    "theta_err 加 2x noise_ratio",
                ],
                "generation_functions": [
                    "src/sim/phased_array_sim.py::simulate (第三级链路预算与服务越限)",
                    "src/sim/phased_array_sim.py::_simulate_subdose (第三级, 与 simulate 逐字一致)",
                    "src/sim/phased_array_sim.py::simulate_gan_state (gan_state 路径链路级)",
                ],
            },
        },
        "hi_construction": {
            "channel_level": {
                "input": "sim_v2 (subdose on, dynamics_id=phased_array_subdose_v2)",
                "builder": "src/sim/build_channel_hi.py",
                "label": (
                    "z=max(dR/dR_th, dI/dI_th, dg/d_g_th); hi=clip(z,0,1); "
                    "v2 双字段: rul_ch_windows=EOL_ch−t (删失=观测终点下界, 不做 0.35T 截顶), "
                    "rul_ch_norm=rul_ch_windows/H (H=mission_horizon 统一任务视界=11688 窗, "
                    "模型数值单位, 非物理寿命比例); 失效通道截断到 EOL (P0-1)"
                ),
                "output": "channel_level.feature_path (canonical 4 维 device_canonical_v1)",
            },
            "service_level": {
                "input": "sim_v1 (subdose off, dynamics_id=leo_coupled_v1)",
                "builder": "src/sim/build_array_hi.py",
                "label": "HI=max(clip(HI_M,HI_SLL,HI_theta)); 右删失 event_observed; D_EOL 取 config 预固定",
                "output": "transfer.target_feature_path (x_global 6 维 + x_nodes 16x6)",
            },
        },
    }
    return chain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phased_array.yaml"))
    ap.add_argument("--out", default=str(ROOT / "docs/simulation/PHYSICS_CHAIN.yaml"))
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    chain = build_chain(cfg, config_path)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# 相控阵三级退化物理链冻结文档 (自动生成, 勿手改)\n"
        "# 生成: python scripts/generate_physics_chain.py\n"
        "# 校验: tests/test_simulation_reproduce_docs.py\n"
    )
    out.write_text(header + yaml.safe_dump(chain, allow_unicode=True, sort_keys=False),
                   encoding="utf-8", newline="\n")
    print(f">> PHYSICS_CHAIN.yaml -> {out}")
    print(f"   config_sha256={chain['provenance']['config_sha256'][:16]}...")


if __name__ == "__main__":
    main()
