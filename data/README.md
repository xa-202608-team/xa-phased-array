# data/ — 本地数据工件槽位

本目录是**本地工件槽位**，存放相控阵组件的仿真/特征数据，**不进入 Git**
（`.gitignore` 已忽略 `/data/`，公开泄漏扫描器对受跟踪的 `data/**` 一律判 CRITICAL）。

## 约定

- 目录内容、来源与哈希登记在 `data_manifest.json`（`schema_version=1.1.0`，
  `component=phased_array`；当前 `status=LOCAL_ARTIFACT_REQUIRED`、`entries=[]`，
  待本地重建工件后逐项补登，不得预填编造的哈希或大小）。
- 与 RC 交付 payload 的映射见 `handoff/artifact-map.yaml`（`local: data` →
  `04_数据/phased_array`）。映射只允许相对路径；本地工件不存在时 RC 打包必须
  失败，而不是生成空包。
- 数据输入字段规范见根目录 `INPUT_SCHEMA.md`。

## 如何重建

在仓库根目录执行（固定随机种子，纯 CPU 可跑）：

```bash
bash scripts/run_all.sh --fast
```

该流程会按 `configs/phased_array.yaml` 重建三级退化链仿真数据
（`data/simulated/...`）与特征 H5（`data/features/...`），随后运行 smoke 训练。
重建完成后再把实际工件补登进 `data_manifest.json`。
