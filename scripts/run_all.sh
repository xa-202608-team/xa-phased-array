#!/usr/bin/env bash
# =====================================================================
# 相控阵组件 - 本地一键复现 (无需 Docker)
#
# 用法:
#   bash scripts/run_all.sh --fast      # smoke (合成数据, 验证管线)
#   bash scripts/run_all.sh             # 正式 (200 轨迹, ~50 分钟 CPU)
#
# 数据策略:
#   - mosfet_canonical.h5 (schema_v4) 和预训练 ckpt 随包发布
#   - full/smoke 模式下 P1/P2 在数据/ckpt 已存在时自动跳过
#   - 双仿真集: sim_v2 (通道级) + sim_v1 (服务级消融)
# =====================================================================
set -euo pipefail
FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1
cd "$(dirname "$0")/.."

CFG="configs/phased_array.yaml"
SM=""
[[ $FAST -eq 1 ]] && SM="--smoke"
CANONICAL_H5="data/features/phased_array/schema_v4/source/mosfet_canonical.h5"
CKPT="checkpoints/source_phased_array_tcn_pretrain.pt"

step() { echo ""; echo "========== $1 =========="; }

# 1. 源域特征工程
step "P1 源域特征工程 (NASA MOSFET)"
if [[ -f "$CANONICAL_H5" ]]; then
  echo "  >> 跳过：$CANONICAL_H5 已存在 (随包发布)"
else
  python -m src.data.preprocess.mosfet_features --config $CFG --synthetic --report
fi

# 2. 源域预训练
step "P2 源域预训练 (TCN 双头)"
if [[ -f "$CKPT" ]]; then
  echo "  >> 跳过：$CKPT 已存在 (随包发布)"
else
  if [[ $FAST -eq 1 ]]; then
    python -m src.train.pretrain --config $CFG $SM
  else
    python -m src.train.pretrain --config $CFG --canonical
  fi
fi

# 3. 相控阵仿真 (双仿真集)
if [[ $FAST -eq 1 ]]; then
  step "P3a 相控阵仿真 sim_v2 (smoke)"
  python -m src.sim.phased_array_sim --config $CFG --fast --seed 42 --subdose on
  step "P3b 相控阵仿真 sim_v1 (smoke)"
  python -m src.sim.phased_array_sim --config $CFG --fast --seed 42 --subdose off
else
  step "P3a 相控阵仿真 sim_v2 (200 轨迹, 子阵级独立损伤)"
  python -m src.sim.phased_array_sim --config $CFG --n_traj 200 --seed 42 --subdose on
  step "P3b 相控阵仿真 sim_v1 (200 轨迹, 旧标量, 服务级消融)"
  python -m src.sim.phased_array_sim --config $CFG --n_traj 200 --seed 42 --subdose off
fi

# 4. HI 构造
step "P4 通道级 HI 构造 (读 sim_v2)"
python -m src.sim.build_channel_hi --config $CFG --report

step "P4b 阵列级 HI 构造 (读 sim_v1)"
python -m src.sim.build_array_hi --config $CFG --report

# 5. 对比实验
step "P5 基线与对比实验"
python -m src.experiments.run_groups --config $CFG $SM --level channel

# 6. 测试
step "P6 测试套件"
python -m pytest tests/ -q --tb=short

step "完成"
echo ">> 全流程结束。结果见 checkpoints/all_metrics_phased_array.json"
