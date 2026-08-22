# F1-A 任务 9：全套测试 + v2 H5 重建验证

## 背景

F1-A 修正门步骤 1-8 已完成：RUL v2 双字段（`rul_ch_windows`+`rul_ch_norm`）、加载器强校验、消费者迁移、config 拆分、factory 统一、测试 fixture 补齐。你的任务是**验证整体闭合**，不改生产逻辑（发现 bug 记录报告）。

## 9.1 全套 pytest

在组件仓根目录运行：
`/f/anaconda3/envs/pytorch_gpu/python.exe -m pytest tests/ -q`

要求：0 failed。已知/可能的 skip（缺本地大数据、timesfm 等）可接受，但要在报告里列出 skip 数与原因分类。若有 failure，**不要自行乱改测试迁就**：先判断是 (a) 真回归（生产代码问题，记录 BLOCKED 详情）还是 (b) 测试仍引用旧 4088/`rul_ch` 口径（这类可以修测试，但要在报告说明）。

重点关注这些测试文件是否受 F1-A 影响且仍绿：
- `test_channel_hi.py` / `test_channel_seq.py` / `test_channel_baselines.py`
- `test_models_factory.py` / `test_rul_scale_policy.py`
- `test_experiment_hygiene.py` / `test_layerwise_transfer.py` / `test_alpha_soft.py`
- `test_service_rollout.py` / `test_fault_injection.py`
- 飞轮/源域/GaN 相关（不应受影响，确认零回归）

## 9.2 v2 H5 重建与字段验证

1. 检查是否存在 sim_v2 原始数据：`data/simulated/phased_array/sim_v2/seed_42/phased_array_all.h5`。
   - 若存在：运行
     `/f/anaconda3/envs/pytorch_gpu/python.exe -m src.sim.build_channel_hi --config configs/phased_array.yaml --report`
     重建 `data/features/phased_array/schema_ch_v1/target/channel_features.h5`。
   - 若不存在：**不要花时间重跑 200 轨迹仿真**（那是 F2 的事）。改为用 9.3 的程序化构造验证 builder 写盘逻辑，并在报告标注"未重建 H5（缺 sim_v2 数据，留 F2）"。
2. 重建后用一个临时 python 片段打开 h5，断言：
   - 顶层 `channel_label_schema == "channel_label_v2"`、`rul_capped == "false"`、`rul_scale_windows == 11688.0`、`sample_period_s == 21600.0`、`t_dev_unit == "degC"`。
   - 任一子组同时含 `rul_ch_windows` 与 `rul_ch_norm`，且 `np.allclose(rul_ch_norm * 11688, rul_ch_windows)`；**不含** `rul_ch`。
   - `read_channel_label_meta(open h5)` 返回正确 dict。
   - `load_target_channel(path)` 返回的 rul 等于所有子组 `rul_ch_norm` 的拼接（值域 [0,1)）。

## 9.3 程序化 builder 写盘验证（若 9.2 未重建数据则必做）

写一个临时脚本（不入库，放 `/tmp` 或运行后删）：用 `build_channel_labels_v2` + 直接 h5py 构造一个最小 v2 h5（结构同 task-8 fixture），确认 `build_channel_hi.main` 的字段命名/attrs 与 `read_channel_label_meta`/`load_target_channel` 三方一致。若 task-8 fixture 已覆盖此点，引用其测试即可，不必重复。

## 9.4 channel level 单组 smoke（可选，数据在才做）

若重建了 H5 且 GPU/CPU 可跑，运行一个 channel 组单 seed smoke（可手动构造小 config 或用现有 smoke 入口），确认 v2 数据下 run_groups 不崩、factor=1.0 日志出现、RMSE 为合理正数。数据不在就跳过并标注。**不跑 5-seed 全矩阵（那是 F2）。**

## 全局约束
- Python `/f/anaconda3/envs/pytorch_gpu/python.exe`。
- 不改 src/ 生产逻辑；测试若因旧口径失败可修，但报告每条改动。
- 不 commit（controller 审查后统一提交）。
- 不派发子 agent。

## 报告
写 `.superpowers/sdd/f1a-closeout/task-9-report.md`：pytest 完整计数（passed/failed/skipped + skip 原因分类）、H5 重建与否及字段验证结果、9.4 smoke 结果、任何 BLOCKED 项。返回只给：状态、测试计数一行、H5 验证结论、concerns。
