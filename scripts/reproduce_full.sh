#!/usr/bin/env bash
# =====================================================================
# reproduce_full (bash 包装) — 与 reproduce_full.ps1 同参数/退出码/输出目录语义
#   用法: bash scripts/reproduce_full.sh --output outputs/full [--config CFG]
#         [--n-traj 200] [--seeds 5] [--fast]
#   退出码: 0=成功 (REPRODUCE_OK 已写出); 非零=任一主步骤失败 (不吞错)
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
exec "$PYTHON" scripts/reproduce_full.py "$@"
