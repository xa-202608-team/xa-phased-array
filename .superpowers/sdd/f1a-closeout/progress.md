# SDD ledger — plan: F1-A 修正门（GPT 审阅 F1 后的返修，无独立 plan 文件，权威=GPT 审阅十步 + 任务内 brief）

Ruling: 无独立 plan/spec 文件；权威为 GPT 审阅给出的 F1-A 十步序列与本目录 task-N-brief.md。步骤 1-7 由 controller 在会话中直接实现并提交（commit cc8f214，BASE），任务 8-9 走 subagent 实现 + review。

## 环境
- Python: `/f/anaconda3/envs/pytorch_gpu/python.exe`（系统 python 不可用）
- 组件仓根: `.collaboration-workspace/xa-phased-array`
- 分支: `feature/f1a-closeout-rul-v2`（基于 origin/main=c7b7a8e）

## Pre-flight scan
- 任务 8 与步骤 3-7 共享 `channel_dataset.read_channel_label_meta` / h5 字段名 / config 键名 — brief 已逐一列出步骤 1-7 产物的精确接口，implementer 不需猜测。
- test_channel_baselines line 231 的 proto 键名 `rul_max_norm` 与生产代码 `channel_baselines.py:419` 一致（未改名），brief 已据此给出。
- 潜在冲突：`_resolve_rul_scale` 在 v2 生产路径已不被调用，但函数与 test 仍在 — 裁决：保留作 v1/service 兼容回归（不删），brief 8.4 已注明。

## 任务
Task 8: in_progress (BASE cc8f214) — 测试更新 + HDF5 fixture
  - implementer DONE_WITH_CONCERNS: tests/test_channel_hi.py(+199, 8 条 tmp_path fixture + 负例), test_channel_baselines.py(双口径断言), test_models_factory.py(config 驱动测试), test_rul_scale_policy.py(legacy 注释)
  - concern 1: 生产 NameError _CHANNEL_META_REQUIRED 未定义 — controller 核实并修复 (cc8f214 之后工作区, 改为 _CHANNEL_META_V2_REQUIRED); 41 passed
  - concern 2: 5 个 baselines e2e 失败 = 本地大数据 h5 混合态 (顶层 v2 attrs / 子组旧 rul_ch), 需任务9 重建数据, 非测试缺陷
  - controller 顺手: scripts/plot_trajectory.py(服务级旧脚本) 也统一到 factory
  - task reviewer 已派发
