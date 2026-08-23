# F4 预注册：±20% 参数敏感性 OAT + PA6 三类消融（2026-08-23，先于执行落档）

> 范围裁定（用户 2026-08-22）：三消融并入 F4；执行用多进程并行。
> 冻结基线 = 既有 sim_v2 seed42 (200 traj) + v2 channel_features.h5（F2 数字 0.1551±0.0131），
> 本批一切变体数据写入独立目录，**绝不覆盖冻结基线路径**。

## A. 物理参数 ±20% OAT（敏感性）

**协议**：一次一参数（OAT），在冻结 config 基础上单参数缩放 ±20%，其余逐字不动；
每变体重生成 sim_v2 (n_traj=200, seed=42) + 重建通道级 HI；与冻结基线（不重跑，直接用
现有数据）对比。共 10 变体。

| # | 记录名 | config 键 | 基线值 | −20% / +20% |
|---|--------|-----------|--------|-------------|
| 1 | Ea（激活能，Arrhenius） | sim.physics.Ea_eV_range | [0.7, 1.1] | [0.56,0.88] / [0.84,1.32] |
| 2 | 热循环（CM 参考温升） | sim.physics.damage_model.coffin_manson.deltaT_ref_K | 20.0 | 16.0 / 24.0 |
| 3 | 热阻（正反馈强度） | sim.physics.damage_model.rth_feedback.a5 | 0.3 | 0.24 / 0.36 |
| 4 | 寿命离差 | sim.physics.life_scale_sigma | 0.3 | 0.24 / 0.36 |
| 5 | 初始余量 | sim.link_budget.initial_margin_dB_range | [1.5, 2.0] | [1.2,1.6] / [1.8,2.4] |

**测量量**（每变体 vs 基线，全部绝对窗口口径）：
- 服务级（sim h5 真值信号，连续 4 窗越限判据同 config.service_limits）：
  median EOL_svc、轨迹失效率、失效约束构成（M_link / SLL / θ_err 占比）
- 通道级（变体 channel_features.h5）：通道失效率（event 占比）、median EOL_ch
- 物理自查：median margin0（应仅随 #5 变）、median Ea_eV 抽样值（应仅随 #1 变）

**判读**：如实报告相对变化（%），无通过/失败门——目的是量化"结论对仿真参数的敏感面"，
支撑仿真可信度论证；若某参数 ±20% 使失效率/EOL 中位数翻转数量级，如实标注为敏感参数。

**不做**：模型级全矩阵重跑（40 臂 × 11 变体不成比例）；如 Tier-A 出现极端敏感参数，
模型级复核另立预注册。

## B1. full_af vs simplified（完整阵列因子 vs 20log10(1−k/N)）

**事后计算**（冻结基线 sim 数据，不重生成）：对每轨迹，
- EOL_full = 多维真值判据首越限（M_link_true≤0 ∨ SLL_true>−8dB ∨ |θ_err_true|>0.5°，连续 4 窗）
- EOL_simplified = 仅增益口径：M_link_s(t) = margin0 − (−20·log10(1−k_failed(t)/N))，首越 ≤0
**报告**：median ΔEOL（simplified − full，窗口）、两 EOL 的相关/一致性、full 口径下
非 M_link 约束（SLL/θ_err 先导）轨迹占比 = 简化公式系统性漏检的失效模式比例。

## B2. channel_count vs continuous（通道计数 vs 连续幅相）

**特征替换协议**：复制基线 channel h5，将 x_ch 第 0 列（p_drift_norm，连续退化观测量）
替换为该子阵存活通道占比（sim subarray_features[...,6]，计数型；终末 dropout 前近常数）。
ch_target_only_gru × seeds{42,43,44} 训练评估（同 F2 协议、v2 口径）。
**判读**：与 F2 主模型 0.1551±0.0131（full 连续特征）对比——RMSE 恶化幅度即
"连续幅相观测量相对通道计数的信息量"；预期计数特征在终末前无判别力 → 显著恶化。

## B3. telemetry_sparsity 三档（全遥测 / 子阵聚合 / 稀疏遥测）

通道级模型可观测性消融，三档定义（通道级口径预注册）：
- **full**：F2 冻结数字（0.1551±0.0131），不重跑
- **subarray（子阵聚合）**：每子阵 x_ch 替换为该轨迹 16 子阵逐窗均值（各子阵特征相同，
  破坏子阵分辨率/个体判别）
- **sparse（稀疏）**：遥测采样稀疏化——每通道序列按 1/6 抽稀（6h→36h 等效 cadence），
  标签保持绝对窗口数不变（rul_ch_windows 为绝对量，抽稀不改变其物理含义；h5 元数据
  H/sample_period 不变，模型窗 L=64 覆盖 6×物理时长）
ch_target_only_gru × seeds{42,43,44} 每档。
**判读**：两档 vs full 的 RMSE 退化量化遥测分辨率/ cadence 价值。

## 执行

- 变体目录：`outputs/f4_oat/{variant}/`（sim + ch h5 + 测量 JSON）、`outputs/f4_ablation/`（B2/B3 h5 + run_groups 产物）
- 并行：A 的 10 个 sim 以 ≤3 并发子进程跑（独立 python 进程，非共享 executor——规避 F2 期间
  Windows spawn 共享池硬崩教训）；B2/B3 共 9 个 run_groups 臂以 ≤3 并发独立进程跑（各独立
  output-dir/jsonl，不共享 executor）
- 产物落档：docs/results_phased_array_f4.md（A 表 + B1/B2/B3 各一节）+ progress.md 行
