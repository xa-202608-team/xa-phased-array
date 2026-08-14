# XA-202608 Phased Array Component

本私有仓库只承载相控阵组件。当前契约版本：`component-contract-v1.0.0`。
正式组件版本只能由 `@xa-202608-team/integrators` 签发。
源码迁移、测试和 Docker 门禁将在相控阵迁移计划中完成。

## 组件内容

LEO 通信卫星相控阵天线 T/R 组件退化建模与剩余寿命（RUL）预测：

- **三级退化数字孪生**：GaN T/R 器件应力退化 → 阵列方向图/热点 → 链路性能，
  优雅降级建模（`src/sim/`，含 `phased_array_sim.py` / `build_array_hi.py` /
  `build_channel_hi.py` / `subarray_array_twin.py`）。
- **跨域迁移**：源域 GaN MOSFET 公开退化数据预训练 → 目标域遥测派生 HI 上迁移，
  观测层（Adapter + MMD/CORAL）与 HI 动力学层双路径（`src/transfer/`、`src/train/`）。
- **对比实验**：物理外推 + constant/hi_extrap/arrhenius 非学习基线 +
  有/无迁移消融（`src/baselines/`、`src/experiments/run_groups.py`）。

## 仓库结构

```
configs/          组件级 YAML（phased_array.yaml / phased_array_gan.yaml）
src/              仿真、预处理、模型、迁移、实验编排
scripts/          run_all.sh 与可视化/分析脚本
tests/            单元与物理一致性测试（含仓库边界守护测试）
docs/             结果图示与 smoke 结果说明
handoff/          交接说明与 RC 工件映射（artifact-map.yaml）
INPUT_SCHEMA.md   输入遥测字段规范
```

## 本地工件槽位

`data/`、`results/`、`checkpoints/` 为**本地工件槽位**，被 `.gitignore` 忽略、
不入库；重建入口与登记约定见各目录下 `README.md` 与 `handoff/HANDOFF.md`。
仓库边界由 `tests/test_repository_boundary.py` 守护。

## 快速开始

```bash
pip install -r requirements.txt        # torch 由 Dockerfile 单独从 CPU 源安装
python -m pytest tests/ -q             # 单元与物理一致性测试
bash scripts/run_all.sh --fast         # 重建仿真数据 + smoke 训练（固定随机种子）
```

Docker 复现见根目录 `Dockerfile` 与 `scripts/entrypoint.sh`；输入字段约定见
`INPUT_SCHEMA.md`。
