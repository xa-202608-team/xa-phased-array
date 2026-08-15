#!/usr/bin/env bash
# =====================================================================
# reproduce_judge (bash 包装) — 与 reproduce_judge.ps1 同参数/退出码/输出目录语义
#   用法: bash scripts/reproduce_judge.sh --output outputs/judge [--config CFG]
#   退出码: 0=成功 (REPRODUCE_OK 已写出); 非零=任一主步骤失败
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
exec "$PYTHON" scripts/reproduce_judge.py "$@"
