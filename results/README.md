# results/ — 本地结果工件槽位

本目录是**本地工件槽位**，存放实验对比结果（RMSE/PHM/MAE 等），**不进入 Git**
（`.gitignore` 已忽略 `/results/`，公开泄漏扫描器对受跟踪的 `results/**` 一律判 CRITICAL）。

## 约定

- `public_summary.json`：公开冻结摘要（`schema_version=1.1.0`，
  `component=phased_array`）。2026-08-21 起为 `status=VERIFIED_WITH_BOUNDARY_EVIDENCE`
  （迁移边界证据逐项核实后填入，**不得编造数字**）。
- `expected_metrics.json`：复现验证的期望指标门槛，gate 仍 `NOT_ENABLED`
  （按其 gate_reason：正式容差须 multi-seed full 复现且跨环境洁净验证后冻结）。
- 槽位白名单清单登记在 `results_manifest.json`（`schema_version=1.1.0`，
  条目含 size/sha256 与 `rc_payload` 开关，批准进 RC payload 时按裁决逐项翻转；
  `scripts/stage_handoff_payload.py` 装配 staging、
  `scripts/scan_handoff_payload.py --repo-root` 以其为批准集扫描）。
- 参考结果与 RC payload 的映射见 `handoff/artifact-map.yaml`
  （`local: results/reference` → `05_结果/reference/phased_array`）。本地参考
  结果不存在时 RC 打包必须失败，而不是生成空包。

## 如何重建

在仓库根目录执行 `bash scripts/run_all.sh --fast`（对比实验编排见
`src/experiments/run_groups.py`）；结果图示（公开报告用）已随代码库提供在
`docs/figures/`，生成结果的 JSON/日志一律落在本目录、不入库。
