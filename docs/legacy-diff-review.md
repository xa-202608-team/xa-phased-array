# 旧交付 vs 组件仓差异文件审核记录

- **日期**: 2026-08-16
- **分支**: `feature/legacy-artifact-migration`
- **审核人**: RC 交付管线 Task 7（自动化审核）
- **左侧（旧交付）**: `XA-202608_最终交付/03_代码/components/phased_array`（只读快照）
- **右侧（组件仓）**: 本仓 `xa-phased-array`
- **裁定原则**（设计 §9.5）: 代码与基础设施以独立仓库版本为基线，不做目录级覆盖。四类裁定 = `仓库版本为准` / `需移植` / `旧交付独有（仅存档）` / `待用户裁决`。

## 1. 实测差异全集

```bash
diff -rq "$LEGACY/03_代码/components/phased_array" "$PA" \
  --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache --exclude=.private \
  | grep -v "^Only in"
```

实测双侧共有且内容不同的文件 **恰好 4 个**，与设计 §2 基线完全一致（无漂移；第 4 个 smoke 报告路径实测确认为 `docs/results_phased_array_smoke.md`）：

| # | 文件 | 裁定 |
|---|------|------|
| 1 | `Dockerfile` | `仓库版本为准` |
| 2 | `requirements.txt` | `仓库版本为准` |
| 3 | `scripts/entrypoint.sh` | `仓库版本为准` |
| 4 | `docs/results_phased_array_smoke.md` | `仓库版本为准` |

**范围边界**: `diff -rq` 另报 40 条 "Only in" 行（旧交付独有 6 项：`docker/`、`run_groups_gpu.log`、`checkpoints/` 下 4 个 metrics/jsonl；组件仓独有 34 项：契约新增的 `schemas/`、`component/`、`scripts/reproduce_*.py|ps1|sh`、`handoff/`、`outputs/` 等）。按任务约定这些单侧独有工件属 Task 6 工件迁移/登记范畴，不在本审核范围内。

## 2. Dockerfile — `仓库版本为准`

**差异要点**（旧交付 → 组件仓）：

1. **工件边界（契约 v1.1 核心变更）**: 旧交付 `COPY checkpoints/` 与 `COPY data/` 把 canonical H5（876KB）+ 预训练 ckpt 烘焙进镜像层；组件仓不拷贝任何大工件，改为运行时从 `/artifacts/data`、`/artifacts/checkpoints` 只读挂载（缺失自动 fallback synthetic 源域，manifest 标 `source_mode=synthetic`），并声明 `VOLUME ["/outputs"]` 唯一可写输出目录。
2. **新增构建参数**: `TORCH_EXTRA_INDEX`（默认 cu130，可切 CPU 变体 `--build-arg TORCH_EXTRA_INDEX=https://download.pytorch.org/whl/cpu`）与 `PIP_INDEX`（清华镜像）；torch 的纯 Python 依赖先从主镜像源装好，避免 pip 经 extra-index 解析落到容器内常不可达的 PyPI 官方域名。
3. **新增 `XA_GIT_COMMIT` 构建参数 + ENV**: 容器内无 `.git`，烘焙源码 commit 供 `reproduce_judge/full` 的 manifest 使用（不烘焙则 judge 因拒绝伪造而失败）；放在依赖层之后，改变 commit 不击穿 pip 层缓存。
4. **构建时验证增强**: 在导入检查 + pytest 之前增加 8 个契约 Schema（`schemas/*.schema.json`）快照校验；pytest 去掉 `-x` 改为 `-p no:cacheprovider`。
5. **拷贝集变更**: 以 `schemas/`、`component/`、`docs/` 替代 `data/`、`checkpoints/`。

**裁定依据**: 组件仓版本是旧交付的严格演进——旧交付"工件随镜像烘焙"模式与契约 v1.1 工件边界直接冲突，属有意淘汰；镜像源容错、CPU 变体、Schema 校验、commit 烘焙均为组件仓新增能力。**旧交付侧不存在组件仓缺失的实质修复**。旧交付注释引用的 `docker-compose.yml` 位于旧交付 `docker/` 目录（"Only in" 集合，Task 6 范畴），组件仓 Dockerfile 注释已同步改为 `docker run` 形式，自洽。

## 3. requirements.txt — `仓库版本为准`

**差异要点**: 唯一差异为组件仓追加两行——注释（契约 v1.1 时序输入与复现输出 Schema 校验）与 `jsonschema==4.25.1`。该依赖被 `component/`、`reproduce_judge.py`、`reproduce_full.py` 及 Dockerfile/entrypoint 的 Schema 快照校验实际使用。

**裁定依据**: 组件仓是旧交付的严格超集，无任何行被删改；旧交付侧无独有内容。无需移植。

## 4. scripts/entrypoint.sh — `仓库版本为准`

**差异要点**（旧交付 → 组件仓）：

1. **子命令重构（契约 v1.1）**: `verify | reproduce_judge | reproduce_full | reproduce（v1.0 兼容） | <任意命令>`；旧交付的 `reproduce smoke|full` 内联流水线全部下沉到 `scripts/reproduce_judge.py` / `scripts/reproduce_full.py`。
2. **`mount_artifacts()`**: 只读挂载点 `/artifacts/data/mosfet_canonical.h5`、`/artifacts/checkpoints/source_phased_array_tcn_pretrain.pt` 存在则复制进工作区，缺失不报错、judge/full 走 synthetic 源域且 manifest 如实标注（不得与正式源域结果混用）。
3. **吞错模式修复**: 旧交付 smoke P5 的 `2>&1 || { echo "[预期跳过]..." }` 与 full P6/P7 的 `|| echo "非阻塞，跳过"` 会吞掉真实失败；组件仓 entrypoint 及 reproduce 脚本明确"任何主步骤失败必须非零退出"。
4. **verify 增强**: 增加契约 Schema 快照校验（8 个），pytest 加 `-p no:cacheprovider`。

**旧内联步骤承接核查**（确认无功能遗漏）:

| 旧 entrypoint 步骤 | 组件仓承接 |
|--------------------|-----------|
| P1 源域特征（canonical 随包/合成） | `scripts/prepare_source_data.py` |
| P2 源域预训练 | `src.train.pretrain --canonical`（ckpt 已存在则跳过） |
| P3a/P3b 仿真 sim_v2/sim_v1（--subdose on/off） | `scripts/generate_simulation.py`（统一入口，循环生成 sim_v2 + sim_v1） |
| P4/P4b 通道级/阵列级 HI | `src.sim.build_channel_hi` / `src.sim.build_array_hi` |
| P5 对比实验 | `src.experiments.run_groups --level channel`（5 seeds） |
| P6 通道级基线 | `src.baselines.channel_baselines`（旧吞错在此修复：失败即失败） |
| P7 结果合并（`merge_matrix_results.py`） | `reproduce_full.py` 内置指标汇总 + metrics/manifest Schema 校验（替代独立合并脚本调用） |

**裁定依据**: 组件仓版本是契约 v1.1 的完整重构，覆盖旧内联流水线全部步骤且修复了吞错缺陷；依赖（`reproduce_judge.py`、`reproduce_full.py`、`schemas/`×8、`component/`）在仓内均存在。**旧交付侧不存在组件仓缺失的实质修复**。

## 5. docs/results_phased_array_smoke.md — `仓库版本为准`

**差异要点**: `diff` 报全文件 27 行差异，但 `diff --strip-trailing-cr` 后**逐字节一致**——纯行尾符差异：旧交付为 CRLF（27 个 CR），组件仓为 LF（0 个 CR）。

**裁定依据**: 组件仓 `.gitattributes` 强制 `* text=auto eol=lf`（跨 OS 仿真一致性），LF 为仓内规范形态；数值表（全 0.0000 的 smoke 占位指标）、验收与实现说明文本两侧完全相同，无任何内容信息丢失。无需移植。

## 6. 结论汇总

- **4/4 `仓库版本为准`**，0 个 `需移植`，0 个 `旧交付独有（仅存档）`，0 个 `待用户裁决`。
- 四个差异全部是"组件仓在契约 v1.1 演进中领先旧交付快照"的方向性差异，不存在旧交付侧实质修复落后于组件仓的情况，**无需开移植 issue**。
- 本记录仅作审核结论，不执行任何覆盖或移植；旧交付快照的工件级登记见 Task 5/6 产物（`scripts/audit_legacy_delivery.py`、`data/data_manifest.json`、`checkpoints/checkpoint_manifest.json`）。
