# Phased Array 输入遥测约定

> 本文件定义相控阵天线组件的输入遥测字段规范。校验器 `scripts/validate_input_schema.py`
> 按此约定对 CSV 输入做字段存在性、单位一致性与数据类型检查。
>
> 数据来源：GaN T/R 组件三级退化数字孪生仿真（应力 → 器件参数漂移 → 阵列性能）。
> 参见 `configs/phased_array.yaml` 与 `src/sim/build_channel_hi.py`。

## 目标域 — 通道级 canonical 特征 (device_canonical_v1)

通道级迁移路线主线使用的 4 维输入（`channel_level.feature_names`）：

| 字段 | 单位 | 采样率 | 必需 | 说明 |
|------|------|--------|------|------|
| `p_drift_norm` | — (无量纲) | 6 h | 是 | 归一化器件参数漂移 = max(ΔI_DSS/δI, ΔP/δP)；源/目标同语义迁移核心量 |
| `T_dev_C` | °C | 6 h | 是 | 器件结温（Tj），Arrhenius 热应力驱动量 |
| `duty` | — (无量纲) | 6 h | 是 | PA 占空比，电热应力条件变量（范围 0.3–0.8） |
| `drive_norm` | — (无量纲) | 6 h | 是 | 归一化驱动幅值 = amp_rms/amp_rms_0，射频激励强度 |

### 失效阈值参数（用于标签派生，非直接输入特征）

| 字段 | 单位 | 说明 |
|------|------|------|
| `R_DS` | — (无量纲) | GaN R_DS(ON) 退化规范阈值 = 0.35 |
| `I_DSS` | — (无量纲) | I_DSS 下降阈值 = 0.20 (≈1 dB 增益跌落) |
| `g_m` | — (无量纲) | 跨导下降阈值 = 0.15 |
| `P_out` | — (无量纲) | 功率漂移阈值 = 0.20 (≈1 dB) |

## 目标域 — 服务级特征 (legacy / cross_level)

旧路径（`model.n_features_target: 12`）的 6 维全局 + 6 维节点均值池化特征：

| 字段 | 单位 | 采样率 | 必需 | 说明 |
|------|------|--------|------|------|
| `M_link_dB` | dB | 6 h | 否 | 链路余量 M_link(t) = EIRP+Gr−L_path−L_other−(C/N0)_req |
| `SLL_dB` | dB | 6 h | 否 | 第一旁瓣电平 (Sidelobe Level) |
| `theta_err_deg` | deg | 6 h | 否 | 波束指向误差 |
| `EIRP_norm` | — (无量纲) | 6 h | 否 | 归一化等效全向辐射功率 |
| `link_margin` | — (无量纲) | 6 h | 否 | 归一化链路余量 |
| `HI_array` | — (无量纲) | 6 h | 否 | 阵列系统级健康指标 |

### 子阵节点特征（16 个子阵 mean-pool 后 6 维）

| 字段 | 单位 | 采样率 | 必需 | 说明 |
|------|------|--------|------|------|
| `mean_pow` | — (归一化) | 6 h | 否 | 子阵平均功率 |
| `q10_pow` | — (归一化) | 6 h | 否 | 子阵功率 P10 |
| `IDSS` | — (归一化) | 6 h | 否 | 子阵 I_DSS |
| `Tj` | °C | 6 h | 否 | 子阵结温 |
| `amp_rms` | — (归一化) | 6 h | 否 | 子阵幅值 RMS |
| `phase_rms` | deg | 6 h | 否 | 子阵相位 RMS |

## 源域 — NASA MOSFET/IGBT 特征 (schema_v3, 5 维)

源域预训练使用的电参数特征（`source.feature_names`）：

| 字段 | 单位 | 采样率 | 必需 | 说明 |
|------|------|--------|------|------|
| `RDS_drift` | — (无量纲) | 老化步级 | 是 | 温度校正后 ΔR_DS(ON) 归一化增量 (NASA Celaya 判据) |
| `T_case_C` | °C | 老化步级 | 是 | 器件外壳温度 |
| `supply_V` | V | 老化步级 | 是 | 供电电压（工况条件化） |
| `gate_voltage` | V | 老化步级 | 是 | 栅极驱动电压（工况条件化） |
| `duty_cycle` | — (无量纲) | 老化步级 | 是 | 占空比（工况条件化） |

> **采样率说明**：源域 NASA MOSFET/IGBT 数据为老化步级采样（非等时间间隔），
> 与目标域 6 h 匀均遥测不同。迁移发生在 HI 动力学层，非原始波形层。

## 仿真参数（configs/phased_array.yaml 关键值）

| 参数 | 值 | 说明 |
|------|------|------|
| 遥测采样周期 | 21600 s (6 h) | `sim.sample_period_s` |
| 物理积分步 | 300 s (5 min) | `sim.physics_dt_s`（分离于遥测以解析 90 min 轨道热周期） |
| 仿真年限 | 8.0 年 | `sim.duration_years` |
| 阵列规模 | 16×16 = 256 元 | `sim.array.n_elements` |
| 子阵块 | 4×4 = 16 子阵 | `sim.array.subarray_block` |

## 数值约定

- `p_drift_norm` ≥ 0（归一化漂移量），HI = clip(z, 0, 1) 其中 z = max(dR/δR, dI/δI, dg/δg)
- `T_dev_C` 典型范围 90–140 °C（GaN HEMT 结温循环范围）
- `duty` 范围 0.3–0.8（PA 占空比）
- `RDS_drift` 源域失效阈值 = 0.05 (NASA Celaya)
- 源域特征为老化步级采样（不等间隔），目标域为 6 h 匀均采样
