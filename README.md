# XA-202608 Phased Array Component

本仓库承载相控阵组件（LEO 通信卫星相控阵天线 T/R 组件退化建模与 RUL 预测）。
当前契约版本：`component-contract-v1.1.0`（Schema 快照在 `schemas/`，与契约 Tag
逐字节一致，`tests/test_component_predict.py` 守护）。
正式组件版本只能由 `@xa-202608-team/integrators` 签发。

> **当前定位（v0.3.0 候选）**：主模型 = `ch_target_only_gru` + 三级退化数字孪生
> （F2 v2 正式 5-seed：RMSE `0.1551±0.0131` ÷H，H=11688 统一任务视界）。
> 公开结论 = `NO_CONFIRMED_POSITIVE_TRANSFER`：全监督下源初始化/MMD 无确认增益
> （三项归因 CI 均跨 0）；k=3 少样本源初始化有害属**少样本边界证据**，不得外推为
> 全监督结论。契约机器枚举 `NO_POSITIVE_TRANSFER_SUPPORTED` 与本公开口径的映射
> 见 `docs/SIMULATION_REPRODUCE.md` §2。

## 组件内容

- **三级退化数字孪生**（差异化主卖点）：GaN T/R 器件应力退化 → 阵列方向图/旁瓣/
  指向 → 链路余量与服务越限，优雅降级建模；结构冻结文档
  `docs/simulation/PHYSICS_CHAIN.yaml`（`scripts/generate_physics_chain.py` 从
  config 导出，测试守护）。
- **跨域迁移结论（冻结）**：`NO_CONFIRMED_POSITIVE_TRANSFER` —— 全监督下源权重
  不可区分于随机初始化（init/full/MMD 三项归因 CI 均跨 0），k=3 少样本源初始化
  显著有害属少样本边界证据；迁移适用边界与负迁移诊断本身为论证内容
  （`docs/MODELING.md` §6，含 F2 v2 5-seed 冻结数字与三项归因 CI）。
- **单指标遥测入口**：`python -m component.predict`（method=
  `causal_single_telemetry`），未知在轨数据接入 + 因果趋势基线，与 PyTorch
  主模型（`src/experiments/run_groups.py`）指标口径严格区分（`docs/MODELING.md` §5）。

## 仓库结构

```
configs/          组件级 YAML（phased_array.yaml / phased_array_gan.yaml）
src/              仿真、预处理、模型、迁移、实验编排
component/        契约 v1.1 单指标预测入口（io 标签隔离 / predictor 因果外推）
schemas/          契约 Schema 快照（component-contract-v1.1.0 Tag 逐字节一致，8 个）
scripts/          统一入口五件套 + entrypoint.sh + 可视化/分析脚本
tests/            单元与物理一致性测试（含仓库边界守护、Schema 快照、三级链冻结）
docs/             MODELING / DATA_DICTIONARY / SIMULATION_REPRODUCE + figures + smoke 结果
handoff/          交接说明与 RC 工件映射（artifact-map.yaml）
INPUT_SCHEMA.md   输入遥测字段规范
```

## 本地工件槽位

`data/`、`results/`、`checkpoints/` 为**本地工件槽位**，被 `.gitignore` 忽略、
不入库；canonical H5 与 `.pt` 权重一律不烘焙进镜像。三个公开描述 Manifest
（`data/data_manifest.json`、`checkpoints/checkpoint_manifest.json`、
`results/public_summary.json`）已登记来源、角色与公开指标摘要（逐项注明出处）。
仓库边界由 `tests/test_repository_boundary.py` 守护。

## 快速开始

```bash
pip install -r requirements.txt        # torch 由 Dockerfile 单独从 CPU/CUDA 源安装
python -m pytest tests/ -q             # 单元/物理一致性/契约/三级链冻结测试

# 评审用小规模端到端（仿真→HI→单指标预测/基线→对比实验→Schema 校验，CPU 分钟级）
bash scripts/reproduce_judge.sh --output outputs/judge    # 或 scripts/reproduce_judge.ps1

# 完整复现（200 轨迹 + 5 seeds；主步骤失败非零退出）
bash scripts/reproduce_full.sh --output outputs/full      # 或 scripts/reproduce_full.ps1

# 单指标遥测预测（输出 prediction.json，符合 schemas/prediction.schema.json）
python scripts/predict_telemetry.py --telemetry T.csv --metadata M.yaml \
    --telemetry-name array_gain_db --output outputs/predict
```

产物：`manifest.json` / `metrics.json` / `run.log` / `REPRODUCE_OK`
（+ `source_mode.txt`；无真实源域时 `source_mode=synthetic`，结果仅证明管线连通，
不得与正式源域结果混用）。

## Docker

```bash
docker build --build-arg XA_GIT_COMMIT=$(git rev-parse HEAD) \
    -t xa-phased-array:v0.3.0-rc.1 .
#（无 GPU / NVIDIA 源不可达时加 --build-arg TORCH_EXTRA_INDEX=https://download.pytorch.org/whl/cpu）
docker run --rm xa-phased-array:v0.3.0-rc.1 verify
docker run --rm -v "$PWD/outputs:/outputs" \
    -v "$PWD/artifacts/data:/artifacts/data:ro" \
    -v "$PWD/artifacts/checkpoints:/artifacts/checkpoints:ro" \
    xa-phased-array:v0.3.0-rc.1 reproduce_judge /outputs/judge
```

`/artifacts/data`、`/artifacts/checkpoints` 只读挂载（`mosfet_canonical.h5`、
`source_phased_array_tcn_pretrain.pt`），缺失自动走 synthetic 源域；
`/outputs` 唯一可写目录。`verify` 只跑代码/fixture/Schema/导入检查。

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/MODELING.md` | 建模方法、HI/标签派生、迁移否定结论（冻结数字与出处） |
| `docs/DATA_DICTIONARY.md` | 数据来源/单位/采样/缺失/划分/标签派生/预测可获得性 |
| `docs/SIMULATION_REPRODUCE.md` | 种子、五件套命令、judge/full 产物、Docker、已知平台问题 |
| `docs/simulation/PHYSICS_CHAIN.yaml` | 三级物理链冻结定义（gan_tr / array / link） |
| `docs/results_phased_array_smoke.md` | smoke 管线连通性结果（非正式数字） |
| `INPUT_SCHEMA.md` | 输入遥测字段规范 |
