# F1-A 任务 8：更新旧测试 + 新增无大数据 HDF5 fixture

## 背景（你只需知道这些）

相控阵组件正在做 F1-A 修正门（RUL 口径 v2）。步骤 1-7 已完成并提交（BASE=`cc8f214`）：
- H5 v2 现在写**双字段**：`rul_ch_windows`（绝对窗口数，不归一）+ `rul_ch_norm`（=窗口数/H，模型标签）；**不再写旧 `rul_ch`**。
- `src/transfer/channel_dataset.py` 新增 `read_channel_label_meta(f)` 强校验 h5 attrs（`channel_label_schema` 必须为 `channel_label_v2`/`channel_label_v1`；v2 必填 `rul_scale_windows/sample_period_s/rul_capped/mission_horizon_windows`，且 `t_dev_unit='degC'`，缺失即 `ValueError`，不 warning）。常量 `CHANNEL_LABEL_SCHEMA_V2="channel_label_v2"`、`CHANNEL_LABEL_SCHEMA_V1="channel_label_v1"` 已导出。
- `load_target_channel` 按 schema 读字段：v2 读 `rul_ch_norm`（已归一，调用方不要再除）；v1 读 `rul_ch`。
- `src/baselines/channel_baselines.py` 的 `_load_channels` 按 schema 读字段，但返回 dict 的 **key 仍叫 `rul_ch`**（内容为绝对窗口数）；`evaluate_channel_baselines` 的 `rul_max` 从 h5 meta 读（v2=H=11688，v1=4088），存在 `proto["rul_max_norm"]`。
- config `configs/phased_array.yaml` 新增 `model.gru.{hidden:64,num_layers:2,dropout:0.1}` 和 `service_level.{rul_scale_windows:4088,rul_label_schema:service_label_v1}`；`channel_level.early_observation_fraction` 和 `channel_level.rul_cap_ratio` 已删除。
- `src/models/factory.py` 的 `build_transfer_model` 现从 `cfg["model"]["gru"]` 读 gru_hidden/layers/dropout 并透传给 `TransferModel`（后者新增了这三个 kwargs，默认 64/2/None）。

## 你的任务（测试侧，不改生产代码逻辑；如发现生产 bug 记录在报告里不要自行大改）

### 8.1 新增最小临时 HDF5 fixture 测试（最重要，解决"clean checkout 无见证"）

在 `tests/test_channel_hi.py` 新增测试，用 `tmp_path` 构造一个**最小合法 v2 channel_features.h5**（不依赖本地 7.8GB 大数据），覆盖：
1. `read_channel_label_meta` 对合法 v2 h5 返回正确字段（schema/rul_scale_windows/sample_period_s/rul_capped=False/t_dev_unit=degC）。
2. v2 h5 同时含 `rul_ch_windows` 和 `rul_ch_norm` 两个 dataset，且 `np.allclose(rul_ch_norm * H, rul_ch_windows)`；**不含** `rul_ch`（断言读 `rul_ch` 会 KeyError，防止原地改语义回潮）。
3. `load_target_channel` 在该 fixture 上能跑通，返回的 `rul` 等于 `rul_ch_norm`（已归一，值域 [0,1)），shape 与 x/hi 一致。
4. 强校验负例：缺 `channel_label_schema` attr → ValueError；`channel_label_schema` 为未知值 → ValueError；v2 缺 `rul_scale_windows` → ValueError；`t_dev_unit != 'degC'` → ValueError；`rul_capped != 'false'` → ValueError。每个负例一个小的临时 h5（或同一 h5 改 attr 后重开）。

fixture 构造要点（参考 `src/sim/build_channel_hi.py` 的 main 写盘结构）：
- 顶层 attrs：`dynamics_id`、`canonical_schema="device_canonical_v1"`、`t_dev_unit="degC"`、`t_dev_conversion`、`channel_label_schema="channel_label_v2"`、`rul_capped="false"`、`rul_scale_windows=11688.0`、`sample_period_s=21600.0`、`mission_horizon_windows=11688.0`、`delta_thresholds="R_DS=0.35,I_DSS=0.2,g_m=0.15,P_out=0.2"`。
- 至少 1 个 group `traj_000`，attr `traj_id` 不需要但子组 `sub_00` 需要：dataset `x_ch`(T,4) float32、`hi_ch`(T,)、`z_ch`(T,)、`rul_ch_windows`(T,)、`rul_ch_norm`(T,)；子组 attrs：`event_observed`(0 或 1)、`eol_idx`、`traj_id=0`、`sub_id=0`、`feature_names`、`canonical_schema`。
- T 取小值（如 128）；失效子组可造一条：`z=np.linspace(0,1.2,T)`，`eol=np.argmax(z>=1)`，`rul_windows=np.maximum(eol-np.arange(T),0)`，`rul_norm=rul_windows/11688.0`。

### 8.2 修正 `tests/test_channel_baselines.py` 的旧 4088 断言

- line 231 附近 `assert proto["rul_max_norm"] == float(cfg["transfer"]["rul_max_norm"])`：v2 下 `rul_max_norm` 来自 h5 meta（=11688），不再是 transfer.rul_max_norm(4088)。改为：用 `read_channel_label_meta` 读 feature_path 的 schema，v2 断言 `proto["rul_max_norm"] == meta["rul_scale_windows"]`；v1 才断言 == 4088。本机现有 h5 是 v2（若存在），所以这条应实际走到 v2 分支。
- line 7 docstring "rul_max_norm 固定归一" 更新为反映 v2 H / v1 4088 双口径。
- line 199 docstring 同样更新。
- 其余 `_z_extrap_predict(..., rul_max=1000.0)` 等单元测试用的是传入的任意 rul_max，与 h5 无关，**不改**。
- line 209/215/216 检查 dict key `rul_ch`：`_load_channels` 返回 dict key 仍叫 `rul_ch`（窗口数），这些断言**仍正确，不改**；但可加一条断言 v2 下其值域是窗口数（>1 可能），非归一值。

### 8.3 更新 `tests/test_models_factory.py`

- 顶部 docstring "GRU 架构参数 (hidden=64/layers=2) 目前依赖 adapter.TRANSFER 默认" 已过时：现已从 `config.model.gru` 显式读取。更新 docstring。
- 现有 `test_factory_gru_hidden_layers_match_adapter_default` 仍通过（值仍 64/2），但请**新增一条测试**：临时改 cfg 的 `model.gru.hidden=128` / `num_layers=1`，构建模型，断言 `model.encoder.gru.hidden_size==128`、`num_layers==1`——证明 factory 从 config 读取而非硬编码。用 `copy.deepcopy(cfg)` 避免污染其他测试。
- `test_factory_roundtrip_weights_with_run_groups_model` 仍应通过（factory 是 run_groups._build_model 的唯一实现）。

### 8.4 评估 `tests/test_rul_scale_policy.py`

- `_resolve_rul_scale` 现在**只服务 service 级 fallback 和 channel v1(legacy)**；channel v2 生产路径已不调用它（run_groups 直接用 h5 meta 的 factor=1.0）。
- `test_channel_mission_horizon_no_second_norm` 仍测 `_resolve_rul_scale(level="channel", policy=mission_horizon)` 返回 `(1.0, 11688)`：函数确实仍返回这个值（未删），但该路径不再被 v2 生产代码调用。请在该测试加注释说明"此为 legacy/v1 兼容路径，v2 生产路径走 h5 meta"，测试本身保留（函数未删，回归保护）。
- 如果 `_resolve_rul_scale` 对 channel+mission_horizon 已无任何生产调用方，你可以在报告里指出，但**不要删函数或测试**（service fallback 仍用）。

## 全局约束

- Python 解释器：`F:/anaconda3/envs/pytorch_gpu/python.exe`（bash 里 `/f/anaconda3/envs/pytorch_gpu/python.exe`）。系统 `python` 不可用。
- 跑测试命令：`/f/anaconda3/envs/pytorch_gpu/python.exe -m pytest tests/test_channel_hi.py tests/test_channel_baselines.py tests/test_models_factory.py tests/test_rul_scale_policy.py -v` 从组件仓根目录。
- 不依赖 GPU。不要改 `src/` 生产代码（除非发现明确 bug，在报告里列出，不要自行修复逻辑）。
- 中文注释。
- 提交：所有改动作为一个 commit，信息 `test(F1-A 任务8): 新增 v2 HDF5 fixture + 修正旧 4088 断言 + factory config 驱动测试`。

## 报告

写到 `.superpowers/sdd/f1a-closeout/task-8-report.md`，含：改了哪些文件、新增测试条数、pytest 输出（passed/failed/skipped 计数）、self-review 发现、任何生产代码疑虑。返回时只给：状态(DONE/DONE_WITH_CONCERNS/BLOCKED)、commit hash、一行测试摘要、concerns。
