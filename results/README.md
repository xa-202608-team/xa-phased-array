# results/ — 本地结果工件槽位

本目录是**本地工件槽位**，存放实验对比结果（RMSE/PHM/MAE 等），**不进入 Git**
（`.gitignore` 已忽略 `/results/`，公开泄漏扫描器对受跟踪的 `results/**` 一律判 CRITICAL）。

## 约定

- `public_summary.json`：公开冻结摘要（`schema_version=1.1.0`，
  `component=phased_array`）。当前 `status=NOT_YET_VERIFIED`、`metrics=[]`；
  只有当指标来源逐项核实后才允许填入，**不得编造数字**。
- `expected_metrics.json`：复现验证的期望指标门槛，同样在未核实前保持
  `NOT_YET_VERIFIED` + 空 `metrics`。
- 参考结果与 RC payload 的映射见 `handoff/artifact-map.yaml`
  （`local: results/reference` → `05_结果/reference/phased_array`）。本地参考
  结果不存在时 RC 打包必须失败，而不是生成空包。

## 如何重建

在仓库根目录执行 `bash scripts/run_all.sh --fast`（对比实验编排见
`src/experiments/run_groups.py`）；结果图示（公开报告用）已随代码库提供在
`docs/figures/`，生成结果的 JSON/日志一律落在本目录、不入库。
