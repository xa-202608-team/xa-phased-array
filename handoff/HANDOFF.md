# HANDOFF — 相控阵组件干净基线交接说明

## 仓库定位

`xa-phased-array` 承载 2026 挑战杯 LEO 卫星健康管理项目中**相控阵天线组件**
的干净代码基线：GaN T/R 组件 → 阵列 → 链路三级退化数字孪生、阵列优雅降级
建模与跨域迁移（HI 动力学层 / 观测层双路径）。当前组件契约版本：
`component-contract-v1.1.0`；正式组件版本只能由 `@xa-202608-team/integrators` 签发。

## 本地基线与工件槽位

本仓库**只含代码与公开说明材料**，以下目录为本地工件槽位（`.gitignore` 忽略、
不入库）：

| 槽位 | 登记文件 | 重建入口 |
|------|---------|---------|
| `data/` | `data/data_manifest.json` | `bash scripts/run_all.sh --fast` |
| `checkpoints/` | `checkpoints/checkpoint_manifest.json` | 同上（含源域预训练） |
| `results/` | `results/public_summary.json`、`results/expected_metrics.json` | 同上（对比实验编排 `src/experiments/run_groups.py`） |

槽位边界由 `tests/test_repository_boundary.py` 守护：`data/`、`results/`、
`checkpoints/` 除七个批准描述文件外不得有受跟踪文件；所有
`.pt/.h5/.hdf5/.log` 不得受跟踪。

## artifact-map 与 RC 打包

`handoff/artifact-map.yaml` 定义本地槽位 → RC 交付 payload 的映射：

- `handoff/payload/data` → `04_数据/phased_array`（dataset）
- `handoff/payload/checkpoints` → `03_代码/components/phased_array/checkpoints`（checkpoint）
- `handoff/payload/results/reference` → `05_结果/reference/phased_array`（reference_result）

约束：映射只允许**相对路径**；本地工件不存在时 RC 打包必须**失败**，而不是
生成空包。打包前需将 `data_manifest.json` / `checkpoint_manifest.json` 的
`entries` 补登完整（含 sha256），`public_summary.json` 仅在指标逐项核实后更新。

> `artifact-map` 的 `local` 一律指向 `handoff/payload/` 白名单 staging（一次性、可删除的派生视图）；整槽位直映禁止。RC 必填元数据（`contract_version`/`environment`/`random_seeds`/`commands`）以 `tests/test_handoff_contract.py` 守护，语义见 `xa-integration/tools/build_rc_artifact.py`。

## 交接要点

1. 运行说明：`README.md`（快速开始）与 `INPUT_SCHEMA.md`（输入遥测字段规范）。
2. 依赖：`requirements.txt`（torch 由 Dockerfile 单独从 CPU 源安装）；
   Docker 复现走根目录 `Dockerfile`。
3. 治理：PR 模板（`.github/pull_request_template.md`）要求申报契约版本、
   单指标影响、是否需要 RC、无标签泄漏与公开扫描结论。
4. 公开泄漏扫描：`git ls-files` + 工作树双来源扫描（工具在集成仓库
   `xa-integration`），任何 CRITICAL/HIGH 违规即 gate 失败。
