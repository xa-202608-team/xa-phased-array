# 相控阵天线组件建模说明（MODELING）

> 组件：LEO 通信卫星相控阵天线 T/R 组件退化建模与剩余寿命（RUL）预测。
> 本文回答"建了什么模、标签怎么派生、迁移结论是什么、两条预测路径口径如何区分"。
> 仿真数据字典见 `docs/DATA_DICTIONARY.md`；复现命令见 `docs/SIMULATION_REPRODUCE.md`；
> 三级物理链冻结定义见 `docs/simulation/PHYSICS_CHAIN.yaml`。

## 1. 问题定义

对 16×16（256 通道、4×4 块 → 16 子阵）GaN T/R 相控阵天线，在轨电热应力与辐照
累积导致器件参数漂移 → 阵列方向图退化（增益下降/旁瓣抬升/指向偏差）→ 链路余量
收敛至多维服务越限。建模目标是：

1. 以遥测可观测量（阵列增益、链路余量、旁瓣、指向误差、结温等）预测通道级/服务级 RUL；
2. 刻画"优雅降级"：通道逐步退出下阵列性能的连续劣化（非突发崩溃）；
3. 检验跨域迁移（NASA MOSFET 源域 → GaN 目标域）的适用边界。

## 2. 三级退化数字孪生（差异化主卖点）

实现全部在 `src/sim/phased_array_sim.py`（`simulate` / `_simulate_subdose` /
`simulate_gan_state`），参数唯一来源 `configs/phased_array.yaml`，结构冻结文档
`docs/simulation/PHYSICS_CHAIN.yaml`（由 `scripts/generate_physics_chain.py`
从 config 导出，测试守护）。

### 2.1 第一级 gan_tr：器件级损伤积分

- Arrhenius 热激活 + Coffin–Manson 轨道热循环疲劳 + R_th 正反馈
  （`R_th(s)=R_th0·(1+a5·s)`，末期加速）；
- `dose = duty · (ΔTj/ΔT_ref)^n · exp(-Ea/kB·(1/Tj − 1/Tref))`；
- 二维高斯热点场驱动空间失效聚簇；`subarray_dose.enabled=true` 时每子阵独立
  `Tj_offset_s + life_scale_s` 积分（sim_v2，`phased_array_subdose_v2`）；
- 输出：子阵损伤真值 `latent_sub_damage (T,16)`、元件级 `f_ch (T,256)`。

### 2.2 第二级 array：器件参数漂移 → 阵列方向图

- GaN 参数漂移：`R_DS↑（1.2–1.5×）`、`I_DSS↓（8–12%）`、`g_m↓（4–8%）`、相位漂移
  （独立零均值 + 空间相关梯度，后者主导旁瓣抬升与持续指向偏移）；
- 完整阵列因子方位切面（`_array_pattern`，扫描角 ±45° 随机、半波长间距），
  主瓣抛物线插值精化；旁瓣电平主瓣窗取左右首零点之间；
- 终末 Weibull 通道退出（k/N 辅助 HI）+ 空间聚簇 → **优雅降级**：增益连续下降、
  失效通道逐步增多，而非全阵崩溃。

### 2.3 第三级 link：阵列输出 → 链路与服务真值

- `ΔEIRP(t) = G_array(0) − G_array(t)`，`M_link(t) = margin0 − ΔEIRP(t)`；
- 服务失效多维越限：连续 4 窗（24h）满足
  `M_link ≤ 0 ∨ SLL > −8 dB ∨ |θ_err| > 0.5°`；
- 产出 `label_fail / eol_idx / failed`（服务失效真值）与右删失轨迹。

## 3. HI 构造与标签派生（遥测/标签分离铁律）

标签全部由预处理派生，非原始字段；真值列（`*_true`、`latent_*`、`label_*`）
绝不进入模型输入 `x`。

| 层级 | 构造器 | 标签 |
|------|--------|------|
| 通道级（器件层，主线） | `src/sim/build_channel_hi.py` 读 sim_v2 | `z = max(dR/δR, dI/δI, dg/δg)`；`hi = clip(z,0,1)`；`rul` 封顶 0.35T；失效通道截断到 EOL（P0-1 铁律，丢弃 EOL 后零标签窗根除泄漏） |
| 服务级（层级消融） | `src/sim/build_array_hi.py` 读 sim_v1 | `HI = max(clip(HI_M, HI_SLL, HI_θ))`；`damage_norm = damage/D_EOL`（D_EOL 为 config 预固定物理常数 0.5765，不遍历含 test 的 eol——P0-3）；未失效轨迹右删失（`event_observed=0`，`rul` 为下界） |

个体划分：按轨迹整体划分 train/val/test = 0.15/0.20/0.65（同一退化轨迹绝不跨
split）；源域按器件 leave-one-device-out。评估 RMSE/PHM/MAE 仅统计失效轨迹，
删失轨迹报下界违反率（P0-2）。

## 4. 主模型与对比实验口径（PyTorch）

- 编码器：TCN / GRU 双头（HI 头 + RUL 头），输入目标域遥测特征
  （通道级 canonical 4 维 `device_canonical_v1`；服务级 x_global 6 维 + 16 子阵节点 6 维）；
- 编排：`src/experiments/run_groups.py`，冻结 5 seeds（42–46），8 组对比
  （ch_target_only_tcn/gru、ch_source_pretrain_frozen、ch_source_mmd_physics、
  ch_random_frozen、ch_random_full_finetune、ch_random_nommd、cross_level_transfer）；
- 非学习基线：`src/baselines/channel_baselines.py`
  （constant / z_extrap / arrhenius / similarity_matching / particle_filter）；
- 迁移判别设计：`random_frozen`（判源初始化）、`random_full_finetune`
  （判 S3 全微调下源 ckpt）、`random_nommd`（判 MMD 贡献）三对照构成完整归因链。

## 5. 单指标遥测入口（causal_single_telemetry）——另一条路径，口径不混同

`python -m component.predict`（`component/` 包，契约 v1.1）：

- **用途**：未知在轨数据接入 + 因果趋势基线。输入符合
  `schemas/telemetry.schema.json` 的长表遥测（如 `array_gain_db`，单位 dB，
  退化方向 decreasing）+ `schemas/dataset-metadata.schema.json` 元数据；
- **方法**：`causal_linear_forecast` 只用历史观测做最小二乘线性拟合外推；
  `estimate_rul` 只在斜率朝失效阈值运动时给 RUL，否则返回 None——不使用任何
  未来信息、不加载权重（`--checkpoint` 传入仅做存在性/可加载性校验，失败即
  非零退出）；输出符合 `schemas/prediction.schema.json`；
- **口径声明**：本入口的外推误差是趋势基线性质，**不得**与第 4 节 PyTorch
  主模型在 run_groups 下的 RMSE/PHM/MAE 对比实验指标混同或直接比较。
  主模型是"同分布多特征监督学习"，本入口是"零训练单遥测因果外推"，
  服务于在轨接入的健康趋势研判与基线锚点。

## 6. 跨域迁移结论（冻结）：NO_POSITIVE_TRANSFER_SUPPORTED

> **⚠ 勘误（2026-08-17 迁移结论清零重审）**：本节数字为 50 轨迹时代的旧冻结口径，
> 与最新 200 轨迹矩阵及三项协议缺陷冲突，**"显著负迁移"结论撤回**，待统一协议重跑后再定论：
>
> 1. **最新 200traj 服务级（5 seeds）**：迁移增益 (target_gru − source_mmd) = −0.0017，
>    CI95 [−0.0104, +0.0071] **跨 0**；init_control / full_control / mmd_control 全部跨 0
>    （05_结果/reference/phased_array/01_PA6_服务级主矩阵/all_metrics_phased_array_200traj.json）。
>    正确表述：**未观察到正迁移，源权重无可测增益**（非"显著负迁移"）。
> 2. **通道级旧结论作废**：① ch_* 迁移组的 ρ·L_phys 使用全零 damage 占位（实验污染）；
>    ② "CI 全负"基于 50 轨迹伪重复 CI，seed 级配对 t-CI 实为 [−0.0864, +0.0250] 跨 0（n=3）。
> 3. **混架构归因失效**：source_mmd 历代跑 TCN、主模型 target_only_gru 跑 GRU，
>    跨组比较无法归因源迁移（target_gru − source_mmd 混合了架构差异）。
> 4. 修复见 commit（fix/migration-experiment-clear）：L_phys 无真值即禁用、迁移归因组统一
>    显式 GRU、canonical T_dev_C 统一 °C（旧 Kelvin）。重跑前本节冻结数字仅供历史追溯。
>
> **重跑完成（2026-08-17，200 traj × 5 seeds，统一 GRU）**：预注册判停条款触发——
> 源 ckpt 主归因 (source_mmd − random_full) = +0.0170，CI95 [−0.0231, +0.0571] 跨 0；
> init/mmd control 均跨 0；target_gru − source_mmd = −0.0052 跨 0（旧"显著负迁移"为
> 污染+伪重复+混架构复合假象）。最终口径：**未观察到正迁移，源权重贡献不可区分于
> 随机初始化**（正迁移探索封口）。详见主仓 docs/开发推进计划/transfer_clear_review_and_positive_gain_plan.md §3.4。
>
> **k-shot×源域臂补充封口（2026-08-18，§4c）**：k∈{1,3} × 5 seeds × {random/MOSFET/IGBT/多源}
> 四臂矩阵——k=1 全部 CI 跨 0；**k=3 下 MOSFET 与 IGBT 源初始化均显著更差**（CI 全正）。
> 被否定的精确命题："器件语义 canonical 输入 + 少样本适配 → 通道寿命任务正迁移"。
> 注：本实验为通道级任务，非独立器件级退化/RUL 实验；后者变体中 HI 动力学层（全监督）
> 已测为零增益，HI 层+少样本与 per-element 任务未测（见方案文档 §4c 任务边界表）。

**A3 Layer-wise 迁移定位（2026-08-20，冻结分析）**：嵌套前缀矩阵固定为 R / P1 / P2 /
Full，固定 channel level、k=3、seeds 42–46；训练事实锚定为 commit
`8965e8e4826a56b653aeed2050dd4ae32119a5a2`。其后的 `8c770d6` 仅作 post-run Markdown
“无主比较”标签渲染修正，`analysis.json` 的数值和判读未变，**不是训练重跑**。`depth*` 必须
只按冻结的 42–46 validation mean 选择：

| 臂 | n | validation RMSE 均值 | test RMSE 均值 |
|---|---:|---:|---:|
| R | 5 | 0.28568636 | 0.29248160 |
| P1 | 5 | 0.32654703 | 0.32863832 |
| P2 | 5 | 0.30240330 | 0.32007623 |
| Full | 5 | 0.32858711 | 0.32340968 |

结果为 `depth*=R`。因此没有源层候选，`primary=null`；P1/P2 没有主比较。下表只保留固定
n=5 的描述性 mean/95% CI，不能将方向包装为正迁移，亦不能对 P1/P2 作有益或有害判定：

| 类型 | 配对 | mean | 95% CI |
|---|---|---:|---|
| direct | P1−R | +0.03615672 | [−0.01820913, +0.09052258] |
| direct | P2−R | +0.02759463 | [−0.01075982, +0.06594907] |
| direct | Full−R | +0.03092808 | [+0.01564872, +0.04620744] |
| incremental | P2−P1 | −0.00856210 | [−0.05329801, +0.03617382] |
| incremental | Full−P2 | +0.00333346 | [−0.04510583, +0.05177275] |

`extension_required=false`，不进入 10-seed 扩测；冻结 `verdict=no_source_candidate`。A3 没有
确认性正信号，按判停不重开 A2、进入 RC/交付。主预测模型仍为 `ch_target_only_gru`，以三级
数字孪生和 target-only 预测为主线；本矩阵只定义 MOSFET 源初始化的迁移适用边界与复现指针
（`outputs/layerwise_a3/analysis.json`、`outputs/layerwise_a3/analysis.md`），不改变模型或命令默认配置。

**历史 A1 端点证据（已由 A3 清零重审与最终口径取代）**：以下保留早期 A1 endpoint 的
冻结数字及其当时判读，供追溯 P0 修复后的实验事实；它们不是本组件当前权威结论，也不能与
A3 的 validation-only 选层结果并列为两条现役结论。当前唯一现役结论为上文 A3：
`depth*=R`、`primary=null`、`verdict=no_source_candidate`，未观察到确认性正迁移；这并不把
历史 A1 证据重述为正迁移。

历史 A1 冻结数字（P0 修复后 5 seeds 全量重跑，归一化 RMSE，越低越好；
出处：项目主仓 `docs/开发推进计划/progress.md` "PA6 GPT 三轮审阅 P0-1/2/3 修复"
与"第九次终修"两节；不得用其取代上文 A3 的当前口径）：

| 组 | RMSE（5 seed 均值） | 说明 |
|----|--------------------|------|
| target_only_gru（主模型） | **0.2521** | 无迁移目标域模型 |
| source_mmd_physics（主迁移组） | 0.2847 | 源 ckpt + S3 全微调 + MMD + L_phys |
| random_full_finetune | 0.2603 | 随机初始化 + 同 S3 协议（最优对照） |
| random_nommd | 0.2654 | 判 MMD 归因 |
| random_frozen / source_pretrain_finetune | 0.2590 / 0.2758 | 判冻结协议归因 |
| 基线 constant / arrhenius | 0.3917 / 0.5290 | 非学习基线（constant 为 5 seed 均值；arrhenius 0.5290 为五划分口径，其 5-seed mean 实为 0.5674） |

历史 A1 端点归因链（配对 ΔRMSE 95% CI，当时判读）：

- 迁移增益（target_gru − source_mmd）= **−0.0326，CI [−0.054, −0.011] 全负**
  → 当时记录为显著负迁移（5/5 seed 差）；
- init_control（source_pretrain − random_frozen）= +0.0168，CI [+0.003, +0.030]
  全正 → 当时记录为源初始化显著有害；
- full_control（source_mmd − random_full）= +0.0244，CI [+0.006, +0.043]
  全正 → 当时记录为 S3 全微调下源 ckpt 显著有害；
- mmd_control（random_full − random_nommd）CI 跨 0 → 当时记录为 MMD 无可度量贡献。

历史教训（保留原因）：P0-1 修复前 `rul[eol:]=0` 的 EOL 后零标签窗混入
训练/测试，曾制造第七次"迁移追平 target"假象；修复后该 A1 端点记录为显著负迁移。
**任何迁移定论前必须确认评估无泄漏。**

保留该历史 A1 记录的工程意义：它说明为什么需进行 A3 的清零重审与 validation-only
选层；最终仍以 A3 的无确认性正信号界定 MOSFET 源初始化的适用边界。组件差异化主卖点 =
三级数字孪生 + 优雅降级 +
遥测驱动 target-only 预测（0.2521 vs constant 0.3917，1.55×）。

## 7. 里程碑与已知边界

- 相控阵组件差异化：三级仿真 + 优雅降级（已冻结，PHYSICS_CHAIN.yaml + 测试守护）；
- target-only 预测路线成立（删服务判据量消融 0.2547 ≈ 0.2521，非读判据量离阈值距离）；
- GaN RFALT 空间矩阵（10 seed/300 轨迹）同样未达预注册正迁移门槛（平均
  Δ = −0.0011，CI 全负；逐 seed 仅 seed44 负向，不宣称"稳定有害"，只宣称
  "未观察到正迁移"）；
- 服务层事件 MAE 尚无有效事件样本（当前矩阵的六步窗口未覆盖首次持续越限
  事件），不得写成已完成的服务寿命验证。
