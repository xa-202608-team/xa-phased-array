# F1-A 任务 9 报告：全套测试 + v2 H5 重建验证

状态：**DONE_WITH_CONCERNS**
日期：2026-08-22
执行环境：组件仓根 `.collaboration-workspace/xa-phased-array`，分支 `feature/f1a-closeout-rul-v2`（未切走），Python `/f/anaconda3/envs/pytorch_gpu/python.exe`（CPU）。

## 9.1 全套 pytest

### 重建前（9.1 首跑，H5 为混合态）
```
10 failed, 361 passed, 6 skipped in 107.93s
```
- **10 failures 全为 KeyError `rul_ch_norm doesn't exist`**，分类 = **(b) 数据陈旧/测试依赖重建数据，非生产回归**：
  - `tests/test_channel_baselines.py`×5：`test_load_channels_structure`、`test_evaluate_channel_baselines_full`、`test_split_by_trajectory_consistent`、`test_evaluate_reproducible_same_seed`、`test_similarity_better_than_constant_typically`
  - `tests/test_channel_seq.py`×5：`test_load_target_channel_returns_correct_shapes`、`test_drop_features_reduces_dim`、`test_channel_seq_dataset_groups_by_channel_key`、`test_channel_seq_does_not_cross_channel`、`test_end_to_end_split_and_kshot`
  - 根因：本地 `channel_features.h5` 顶层 attrs 已是 v2 但子组仍为旧单字段 `rul_ch`（混合态），`load_target_channel`（v2 强校验）读 `rul_ch_norm` 必然 KeyError。
- **6 skips 全部为 test_source_io.py 数据可得性 skip**（缺真实源域 h5，需先跑 loader）：
  - 4 条："真实 schema_v2 MOSFET h5 不存在（需先跑 mosfet_real_loader）"（行 24/56/81/105）
  - 2 条："飞轮 schema_v1 source h5 不存在（需先跑 wheel_features）"（行 46/121）

### 重建后（9.1 复跑，v2 H5 + 已修 2 处旧字段引用）
```
371 passed, 0 failed, 6 skipped in 925.75s (0:15:25)
```
- **0 failed 达成**。合计 371+6=377 与重建前 361+10+6=377 对数一致：原 10 失败转绿（+10），`test_channel_hi` 残留旧字段 2 测已修（保持绿），无新增回归。
- 6 skips 与重建前同（test_source_io.py，缺真实源域数据）。
- 注意：重建后 channel e2e 测试完整执行（4 次 `evaluate_channel_baselines`），全套耗时从 ~108s 升至 ~15.5min。只重跑失败 e2e 的隔离验证：`tests/test_channel_baselines.py tests/test_channel_seq.py` → `32 passed in 775.31s`。

## 9.2 v2 H5 重建与字段验证

- **sim_v2 原始数据存在**：`data/simulated/phased_array/sim_v2/seed_42/phased_array_all.h5`（1.7GB），执行了重建：
  ```
  /f/anaconda3/envs/pytorch_gpu/python.exe -m src.sim.build_channel_hi --config configs/phased_array.yaml --report
  ```
  结果：200 轨迹 × 16 子阵 = **3200 通道**，耗时 ~30s，写回 `data/features/phased_array/schema_ch_v1/target/channel_features.h5`（~777MB）。
- **字段断言全部通过**：
  - 顶层 attrs：`channel_label_schema=channel_label_v2`、`rul_capped=false`、`rul_scale_windows=11688.0`、`sample_period_s=21600.0`、`t_dev_unit=degC`。
  - 全部 3200 子组（traj_XXX/sub_00..15）：均同时含 `rul_ch_windows`+`rul_ch_norm`（float32），`np.allclose(rul_ch_norm*11688, rul_ch_windows)` 为真；**旧字段 `rul_ch` 数量 = 0**。
  - `read_channel_label_meta(open h5)` 返回 `{'channel_label_schema':'channel_label_v2','rul_scale_windows':11688.0,'sample_period_s':21600.0,'rul_capped':False,'mission_horizon_windows':11688.0,'t_dev_unit':'degC'}`。
  - `load_target_channel(path)` 返回 9 元组；`rul`（shape (23,910,801,)）== 3200 个子组 `rul_ch_norm` 按序拼接，值域 **[0,1)**。
- 重建报告观测值：通道失效率 55.0%（1760/3200）；median EOL_ch=3052 vs median EOL_svc=2423（见 Concerns）。

## 9.3 程序化 builder 写盘验证

9.2 已实际重建真实 H5，此项非必做。task-8 已自带最小 v2 HDF5 fixture（`tests/test_channel_hi.py::_write_min_v2_h5`）覆盖三方一致性：
- `test_read_channel_label_meta_v2_ok`：meta 读取正确；
- `test_v2_fixture_dual_fields_no_legacy_rul_ch`：双字段 + 无旧 `rul_ch` + `norm*H==windows`；
- `test_load_target_channel_v2_fixture`：loader 返回已归一 rul；
- 强校验负例（缺 schema / 未知 schema / 缺 rul_scale_windows / t_dev_unit≠degC / rul_capped=true → ValueError）。
加上 9.2 真实 H5 的端到端断言，builder 写盘字段命名/attrs 与读侧完全一致。

## 9.4 channel level 单组 smoke

数据在且 CPU 可跑，执行（临时输出目录，不污染仓内 artifacts）：
```
/f/anaconda3/envs/pytorch_gpu/python.exe -m src.experiments.run_groups \
  --config configs/phased_array.yaml --smoke --level channel --groups ch_target_only_gru \
  --output-dir <temp>
```
- 结果：run_groups **不崩**；日志出现 **`[rul-scale] channel v2: factor=1.0, H=11688.0`**；单组输出 `CH Target-only (GRU)`：**RMSE=0.7058**、PHM=18.72、MAE=0.7056、删失违反率=0.000（RMSE 为合理正数，2 epoch smoke + 随机初始化预期值域）。
- results md/json 正常写入临时目录。未跑 5-seed 全矩阵（归 F2）。

## 测试改动记录（类别 b，均已说明）

- `tests/test_channel_hi.py`：
  - `test_failed_channel_truncated_to_eol`：`sub["rul_ch"]` → `sub["rul_ch_norm"]`（v2 已删 `rul_ch`，用模型标签字段；断言语义本就是归一末端=0）。
  - `test_censored_rul_capped`：`sub["rul_ch"]` → `sub["rul_ch_norm"]`（断言本就是 H 归一口径，仅字段名未随 v2 更新）。
  - 改动后该文件 `19 passed`。改前在旧（混合态）H5 下通过、v2 H5 下必 KeyError——属 brief 9.1 允许修复的旧字段引用。

## 生产代码 / BLOCKED

- 未改任何 `src/` 生产代码逻辑；未 git add/commit。
- **无 BLOCKED 项。**

## Concerns

1. **EOL_ch(中位 3052) > EOL_svc(2423)**：重建报告"优雅降级 EOL_ch<EOL_svc=否"。该关系由仿真数据 + δ 阈值决定，非 F1-A 引入（v1/v2 标签计算中 eol 口径一致），但与本仓 §4c"优雅降级"叙述相反，建议物理侧/F2 核对 δ 标定或 `label_fail` 判据（不在 9.1-9.4 断言范围，不影响测试结果）。
2. **全套回归时长**：重建后 channel e2e 完整执行，`pytest tests/ -q` 从 ~108s 增至 ~15.5min。如 CI 跑全套需预留时间预算。
3. smoke 中 `best val_rmse=0.0000`（32 样本 val 子集 + 2 epoch，疑似早停退化到平凡解），仅 smoke 连通性观察，不构成验收问题。