# checkpoints/ — 本地模型权重槽位

本目录是**本地工件槽位**，存放源域预训练与迁移微调产生的模型权重，**不进入 Git**
（`.gitignore` 已忽略 `/checkpoints/`；权重文件另受 `*.pt/*.pth/*.ckpt` 双重忽略）。

## 约定

- 权重清单登记在 `checkpoint_manifest.json`（`schema_version=1.1.0`，
  `component=phased_array`；当前 `status=LOCAL_ARTIFACT_REQUIRED`、`entries=[]`，
  待本地训练产出后逐项补登 sha256，不得预填）。
- 与 RC 交付 payload 的映射见 `handoff/artifact-map.yaml`（`local: checkpoints` →
  `03_代码/components/phased_array/checkpoints`）。本地权重不存在时 RC 打包必须
  失败，而不是生成空包。

## 如何重建

在仓库根目录执行 `bash scripts/run_all.sh --fast`（含源域预训练），或按
`docs`/`scripts` 中对应实验脚本单独重建指定权重；训练配置以
`configs/phased_array.yaml` 为准。
