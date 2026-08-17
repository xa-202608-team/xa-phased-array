# 相控阵组件数据字典（DATA_DICTIONARY）

> 覆盖 FAQ 全项：数据来源 / 单位 / 采样 / 缺失策略 / 划分 / 标签派生 /
> 预测时可获得性。建模方法见 `docs/MODELING.md`；三级物理链结构见
> `docs/simulation/PHYSICS_CHAIN.yaml`；输入字段规范另见根目录 `INPUT_SCHEMA.md`。

## 1. 数据来源总览

| 数据集 | 来源 | 是否随仓库分发 | 生成/下载入口 |
|--------|------|----------------|----------------|
| NASA MOSFET 源域原始（7.85GB MAT，37 case：32 失效/10 删失口径见主仓 progress） | NASA 公开退化数据（MOSFET_Thermal_Overstress_Aging） | 否（不可再分发，不入 Git） | 手动下载 + `src/data/preprocess/mosfet_real_loader_v2.py`（见 §7） |
| 源域 canonical 特征（4 维，`device_canonical_v1`） | 真实 NASA 特征经 `to_canonical_source` 转换；或本地 synthetic 合成 | 否（本地工件槽位 `data/`） | `scripts/prepare_source_data.py` |
| 相控阵目标域仿真 sim_v2（`phased_array_subdose_v2`） | 本地 GaN T/R 三级退化仿真生成（GaN 器件参数为文献量级标定的仿真值，**非真实 GaN 遥测**） | 代码可复现，数据不入 Git | `scripts/generate_simulation.py`（subdose on） |
| 相控阵目标域仿真 sim_v1（`leo_coupled_v1`） | 同上（旧标量损伤路径，服务级层级消融用） | 同上 | `scripts/generate_simulation.py`（subdose off） |
| 通道级特征（`schema_ch_v1`，canonical 4 维） | sim_v2 派生 | 否 | `python -m src.sim.build_channel_hi` |
| 服务级特征（`schema_v1`，6+16×6 维） | sim_v1 派生 | 否 | `python -m src.sim.build_array_hi` |

随机性与种子：仿真固定 seed=42（轨迹参数由 `np.random.default_rng(42)` 采样，
每轨迹独立子流 `seed_traj`）；对比实验冻结 5 seeds = 42..46。相同 seed + 相同
config 的 CPU 仿真跨平台逐位一致。

## 2. 目标域仿真遥测字段（df 列，6h/窗）

采样：`sim.sample_period_s = 21600 s`（6 h 遥测窗，8 年 ≈ 11680 窗）；
物理积分步 `sim.physics_dt_s = 300 s`（正确解析 90 min 轨道热周期）。

| 字段 | 单位 | 预测时可获得 | 派生 | 说明 |
|------|------|--------------|------|------|
| `t` | s | — | 否 | 窗口时间（物理秒） |
| `G_array_dB` | dB | **是** | 否（观测） | 阵列增益观测（加噪）；**单指标入口主遥测 `array_gain_db` 即此列**，退化方向 decreasing |
| `G_array_dB_true` | dB | 否（仅评估） | 否 | 阵列增益真值（无噪，score-only） |
| `M_link_dB` | dB | **是** | 否（观测） | 链路余量观测（加噪） |
| `M_link_dB_true` | dB | 否（仅评估） | 否 | 链路余量真值 |
| `SLL_dB` | dB | **是** | 否（观测） | 第一旁瓣电平观测 |
| `SLL_dB_true` | dB | 否（仅评估） | 否 | 旁瓣真值 |
| `theta_err_deg` | deg | **是** | 否（观测） | 波束指向误差观测（噪声 ×2） |
| `theta_err_deg_true` | deg | 否（仅评估） | 否 | 指向误差真值 |
| `EIRP_norm` | 无量纲 | 否（仅评估） | 否 | 归一化 EIRP（latent，不进 x） |
| `k_failed` | 通道数 | **是** | 是（由 dropout 计数） | 失效通道数（优雅降级辅助 HI） |
| `RDS_drift` | 无量纲（×初始） | **是**（遥测口径） | 是 | 轨迹均值 R_DS(on) 漂移 |
| `IDSS_ratio` | 无量纲 | **是**（遥测口径） | 是 | 轨迹均值 I_DSS 比例 |
| `gm_ratio` | 无量纲 | **是**（遥测口径） | 是 | 轨迹均值跨导比例 |
| `Tj` / `Tj_max` / `Tj_min` | °C | **是** | 否 | 6h 窗结温均值/最大/最小（修复: 仿真器输出 Kelvin 已转为 °C） |
| `duty` | 无量纲（0.3–0.8） | **是** | 否 | PA 占空比 |
| `damage` | 无量纲 | 否（latent） | 是 | 轨迹级标量损伤真值（eff_age/life_scale） |
| `label_fail` | 0/1 | 否（标签） | 是 | 服务失效标记（多维越限+持续判据） |
| `subarray_features` | (T,16,8) | **是** | 是 | 子阵遥测：`[mean_pow, q10_pow, IDSS, Tj, amp_rms, phase_rms, eff_ratio, q90_f]`（修复: Tj 列已转为 °C） |
| `latent_sub_damage` | (T,16) 无量纲 | 否（latent 真值） | 是 | 子阵损伤真值（sim_v2 专属，通道级标签源） |
| `twin_*` | — | 否（twin_only） | 是 | 物理孪生静态量（c_elem/eta_R/eta_phi/dropout_thr/grad_dir/subarray_ids），不进模型输入 |

噪声：`sim.disturbance.noise_ratio = 0.01`（1% 满量程，theta_err ×2）；
遥测缺失 `telemetry_missing_prob = 0`（稀疏档消融时调高）。

## 3. 源域字段（canonical 4 维，`device_canonical_v1`）

| 字段 | 单位 | 说明 |
|------|------|------|
| `p_drift_norm` | 无量纲 | 归一化到器件失效阈值的关键参量漂移：源域 `ΔR_DS/0.05`（NASA Celaya 判据）；与目标域 `max(ΔIDSS/δI, ΔP/δP)` 同语义——迁移接口核心 |
| `T_dev_C` | °C | 器件温度（源域壳温 / 目标域子阵 Tj） |
| `duty` | 无量纲 | 占空比工况 |
| `drive_norm` | 无量纲 | 归一化驱动强度协变量（源域 supply_V×gate_voltage / 初值） |

辅助标签列（源域，`hi / rul_s / rul_lower_bound_s / elapsed_time_s /
event_observed`）：由 `construct_labels` 派生，删失器件 RUL 为下界。

## 4. 缺失策略

- 遥测：任一字段缺失整行剔除，不做插值、不做前向填充
  （契约 dataset-metadata `missing_values.policy=drop`）；
- 仿真数据本身为规则网格、无缺失（`telemetry_missing_prob=0`）；
- 单指标入口（`component/io.py`）：遥测表出现标签通道（`true_rul`、
  `damage_truth` 等值级或列级）直接拒绝，不进入预测。

## 5. 划分策略（防泄漏铁律）

- 目标域：按**轨迹个体**整体划分 train/val/test = 0.15/0.20/0.65
  （`transfer.split`）；同一退化轨迹绝不跨 split；通道级同轨迹 16 子阵同 split
  （`assert_split_by_trajectory` 守护）；
- 源域：按**器件个体** leave-one-device-out（`source.split.method`）；
- 评估：RMSE/PHM/MAE 仅失效轨迹；删失轨迹报 RUL 下界违反率。

## 6. 标签派生（全部为派生字段，非原始观测）

| 标签 | 派生规则 | 派生位置 |
|------|----------|----------|
| 通道级 `z/hi_ch` | `z = max(dR/δR, dI/δI, dg/δg)`，δ = `channel_level.delta_thresholds`（R_DS 0.35 / I_DSS 0.20 / g_m 0.15 / P_out 0.20）；`hi = clip(z,0,1)` | `build_channel_hi.build_channel_labels` |
| 通道级 `rul_ch` | 自当前窗到 z≥1 的窗数，封顶 0.35T；失效后截断（P0-1） | 同上 |
| 服务级 `hi_array` | `max(clip(HI_M, HI_SLL, HI_θ))`，HI=1 即多维服务越限 | `build_array_hi.build_features` |
| 服务级 `damage_norm` | `damage / D_EOL`，D_EOL=0.5765 为 config 预固定物理常数（不遍历含 test 的 eol，P0-3） | 同上 |
| 服务级 `rul` / `event_observed` | 失效=精确 RUL；未失效右删失（event=0，rul 为下界） | 同上 |
| 服务失效 `label_fail/eol_idx` | 连续 4 窗（24h）`M_link≤0 ∨ SLL>−8dB ∨ |θ_err|>0.5°` | `phased_array_sim` 第三级 |

## 7. 正式（非 synthetic）源域获取

`scripts/prepare_source_data.py` 在无任何真实源域文件时自动落
**source_mode=synthetic**（合成 5 维 schema_v3 → canonical），只用于管线
连通性 smoke，结果不与正式源域混用（manifest 如实标注）。需要正式源域：

1. 手动下载 NASA MOSFET Thermal Overstress Aging 原始 MAT（约 7.85GB，S3 断点续传；
   数据页面见 NASA 数据门户，许可 US Government Works 类公共数据）；
2. 运行 `src/data/preprocess/mosfet_real_loader_v2.py` 生成 schema_v3 特征
   `data/features/phased_array/schema_v3/source/mosfet_source_features.h5`；
3. 重跑 `scripts/prepare_source_data.py`（检测到 v3 存在 → 转 canonical，
   source_mode=nasa_real）→ `reproduce_full`。
