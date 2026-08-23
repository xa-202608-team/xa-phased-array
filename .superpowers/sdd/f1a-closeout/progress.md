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
  - task reviewer: Approved (spec 8.1-8.4 全达标, 质量通过), 3 Minor 不入 loop
  - committed 1d97251
  - minor (deferred to 终审/任务9): ①baselines v2 值域断言 np.any(rul>1) 边界理论误报(真数据不会); ②baselines v2 分支断言待任务9真实数据实际走到; ③test_channel_hi loader 测试可加 len(rul)==2*T 显式断言
Task 8: complete (commits cc8f214..1d97251, review clean, 3 minor deferred)
Task 9: complete (commits 1d97251..dcf6460)
  - 重建前 10 failed(混合态h5 KeyError)/361 passed/6 skipped; 重建 v2 channel_features.h5
    (200 traj×16=3200 通道, ~30s) 后全套 **371 passed / 0 failed / 6 skipped** (925s)
  - 字段断言全过: v2 attrs 正确; 3200 子组双字段 allclose; 旧 rul_ch 数=0; loader 一致
  - 改动 tests/test_channel_hi.py 两处存量直读 rul_ch -> rul_ch_norm (brief 允许 b 类)
  - concern(不属F1-A): median EOL_ch>EOL_svc 是跨轨迹 median 误口径(M2 已记正确=逐轨迹 Δ_first)
  - 9.4 smoke 通过 (factor=1.0, H=11688 日志, RMSE 正数)
终审(final review, opus): 无 Critical, 准予进入 F2; 3 Important + 4 Minor
终审修复 (b92bf34): I-1 run_fault eval rul_norm 透传 H / I-2 calibrate .get 兜底 /
  I-3 run_groups channel v1 直接用 rul_max_norm / Minor-1 service 回退 .get / Minor-3 文档双字段
  - 59 passed (子 agent) + 独立补跑 50 passed (experiment_hygiene/service_rollout/layerwise/alpha)
F1-A 完成: 5 commits on feature/f1a-closeout-rul-v2 (merge-base c7b7a8e)
  f5e1df0 (cherry-pick F1 WIP) / cc8f214 (步骤1-7) / 1d97251 (任务8 测试) /
  dcf6460 (任务9 验证) / b92bf34 (终审修复)
  全套 371 passed / 0 failed / 6 skipped; v2 h5 重建(3200 通道)
  未做: F2 正式 5-seed 数字重跑 / F3-F7 (InferenceDataset/bundle/RC/发布)

## 用户裁决 (2026-08-22, F2 开工前)
- 分支不 push、不开 PR, 在 feature/f1a-closeout-rul-v2 上直接续做 F2 (批2 InferenceDataset +
  批3 bundle/predict_gru + 5-seed 正式重跑一体推进)。
- PA6 三类消融 (channel_count/full_af/telemetry_sparsity) **并入 F4** 一起跑, 执行时用
  多线程/多进程并行 (run_groups --workers >1 / 任务级并行), 不再单列。

## F2 执行 (2026-08-22 起)
- 批2: src/transfer/channel_inference.py (ChannelInferenceDataset 无标签/窗口末端/特征校验)
  + schemas/rul-prediction.schema.json
- 批3: run_groups --export-inference-{dir,group,seed} (仅匹配组-seed val-best, channel v2)
  + component/predict_gru.py + reproduce_full 依赖闭合 (P5 导出 → P5.5 推理自检入 metrics)
- 批2/3 完成 (本 commit): 测试 +18 (12 inference dataset + 6 bundle/predict_gru) 全绿;
  全套回归 389 passed / 0 failed / 6 skipped (371 基线零回归);
  真实数据冒烟: 导出不扰动训练 (RMSE=0.7058/PHM=18.72 与任务9.4 逐位一致),
  predict_gru 真实 bundle+h5 整链通过 (norm×H×sp_s 换算核对)。
- 5-seed: 冻结 F2 代码后正式重跑 (workers=2), 作废旧数字

## F2 正式跑完成 (2026-08-22 晚)
- 执行事故: run-1/workers=2 于 27 臂硬崩, run-2 续至 29 臂再崩 (Windows spawn 孤儿 worker,
  父进程死, 与 reproduce_full 注释的退出期崩溃同型; 无 traceback); 清孤儿后 run-3 串行
  (--workers 1, python -u) 收口 40/40 臂, jsonl 零损失。EXIT=127 为 Windows 收尾期退出码
  问题 (产物哨兵齐全, 同 reproduce_full 已知现象)。
- resume 导出守卫: 376006a — 导出臂被 resume 跳过且 bundle 已在盘 → 沿用续跑 (否则拒绝);
  13 测试验证, run-3 实际走到该分支。
- level_control 两缺陷修复 (本轮 commit): ① *_kall 硬编码组名 → 裸名回退 (init/full/mmd
  三对照同步回退); ② 跨层级归一口径混用 → 绝对窗口口径配对, 每臂落盘 rul_scale_windows
  (jsonl 40 条已按 h5 attrs/config 真源回填), 渲染注明单位与标签定义差异; +4 单测
  (test_rul_scale_policy), 受影响子集 49 passed。
- 新数字 (v2 ÷H=11688, 旧 v1 数字作废): gru 0.1551±0.0131 / tcn 0.1927±0.0254 /
  source_frozen 0.1693±0.0151 / source_mmd 0.1693±0.0145 / random_frozen 0.1710±0.0163 /
  random_full 0.1691±0.0148 / random_nommd 0.1476±0.0253 / cross_level(service ÷4088)
  0.2793±0.0245; 归因三对照 CI 全跨 0 → NO_POSITIVE_TRANSFER 冻结结论 v2 口径复现;
  level_control(窗口口径) Δ=−837.6±153.0 CI[−1027.6,−647.5] (service 侧更低, 含任务视界差,
  降级探索性)。产物: outputs/f2_formal_5seed/ (results md + all_metrics json + bundle)。

## F3 manifest 换血 (2026-08-22 深夜, 本 commit)
- results_manifest 重建 (6 RC + 3 归档): F2 all_metrics + 40 臂 jsonl + resume/aggregate 双日志
  + v2 非学习基线 + 边界摘要 (rc_payload=true, verified_local, git=074b318); 历史
  arch_aligned/kshot_frozen/源预训练摘要 → archived_* + rc_payload=false (撤 RC, 不撤文件);
  空日志 run_groups_gpu.log 条目+文件双删。
- v2 基线 (失效子集/同 split/÷H): **z_extrap 0.1588 / similarity 0.1795 / constant 0.2293 /
  arrhenius 0.2945 / particle_filter 0.6833** — z_extrap 落主模型 seed 波动带内 (0.1551±0.0131),
  对模型领先幅度小于波动, 已按诚实口径写入 artifact-map (F5 文档统一需同步此口径)。
- 三处旧叙事修复: artifact-map public_summary (作废 0.2521 + 已撤回"显著负迁移"→ v2 数字 +
  NO_CONFIRMED_POSITIVE_TRANSFER); checkpoint_manifest status_note (同撤回口径) +
  两 ckpt reproduce_status 升级 frozen_artifact_sha_verified_local (sha 与盘上核实一致);
  results/README public_summary 状态行。
- 连带发现并修: data_manifest entry channel_features.h5 sha 为 v1 旧值 (任务9 v2 重建后漏更) →
  更新 sha/size/git/reproduce_status + provenance_note; 其余 14 文件全 OK。
- 验证: stage --clean 20 文件全 sha 对; scan_handoff_payload 0 violations;
  handoff 测试子集 20 passed。staging 残留教训: stage 必须配 --clean (旧 v0.2.0 拷贝会滞留)。

## F4 敏感性 OAT + PA6 三消融 (2026-08-23, 本 commit)
- 预注册先于执行 (eb63718): docs/f4_sensitivity_ablation_design.md — OAT 5参数×±20% +
  B1 full-vs-simplified AF + B2 计数vs连续 + B3 三档遥测; 变体数据全在 outputs/f4_* 不触冻结基线。
- A (10 变体 sim×200traj 3并发 + HI + 测量): 结论稳健 (失效率带 0.53-0.64, EOL 变化≤±12%);
  Ea 唯一显著敏感 (物理预期); **ΔT_ref 精确不变=结构不变性** (剂量尺度被 life_ref 归一吸收,
  life_scale_years 79589/40750/23582 证明参数生效, 非死旋钮); margin0 自洽 (EOL_ch 严格不变)。
- B1 (事后, 基线 sim): 简化公式 median +475 窗系统性偏晚, ρ=0.996 排序保真, 漏检 3.3%,
  本数据 SLL/θ 无先导绑定 (0%) — 能力差异未触发, 如实报告。
- B2: 计数特征 0.3152±0.0106 (+0.160, 2×恶化) — 连续幅相承载预后信息主体。
- B3: subagg 0.1656±0.0043 (+0.011 温和) / sparse 1/6 0.1245±0.0074 (−0.031 反优,
  长上下文>数据量, 协议耦合已注明; cadence 预算 6× 冗余)。
- 执行事故: build_channel_hi 参数为 --in 非 --indir (首轮 10 变体 HI 全败, sim 已在盘修复后
  秒级补齐); run_groups 变体产物名随 config stem (all_metrics_config.json); 0xC0000409
  收尾期退出码以产物哨兵判定 — 三处均改脚本并复跑验证。
- 产物: docs/results_phased_array_f4.md + outputs/f4_oat/oat_results.json +
  outputs/f4_ablation/{b2_count,b3_subagg,b3_sparse}/。
