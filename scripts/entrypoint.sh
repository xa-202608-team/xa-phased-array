#!/bin/bash
# =====================================================================
# 相控阵组件 - 容器入口脚本 (契约 component-contract-v1.1.0)
#   子命令: verify | reproduce_judge | reproduce_full | reproduce | <任意命令>
#
# 工件边界:
#   - canonical H5 / 预训练 ckpt 不烘焙进镜像; 从 /artifacts/data 与
#     /artifacts/checkpoints 只读挂载读取 (缺失时走 synthetic 源域 smoke,
#     输出 manifest 标 source_mode=synthetic, 不得与正式源域结果混用)
#   - /outputs 为唯一可写输出目录
#   - 任何主步骤失败必须非零退出 (旧 `|| echo` 吞错模式已移除)
# =====================================================================
set -euo pipefail

export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

CFG="configs/phased_array.yaml"
CANONICAL_H5="data/features/phased_array/schema_v4/source/mosfet_canonical.h5"
CANONICAL_MOUNT="/artifacts/data/mosfet_canonical.h5"
CKPT="checkpoints/source_phased_array_tcn_pretrain.pt"
CKPT_MOUNT="/artifacts/checkpoints/source_phased_array_tcn_pretrain.pt"

step() { echo ""; echo "========== $1 =========="; }

mount_artifacts() {
    # 只读挂载 -> 复制进工作区 (挂载点只读, 工作区可写); 缺失不报错, 走 synthetic
    if [ -f "$CANONICAL_MOUNT" ]; then
        mkdir -p "$(dirname "$CANONICAL_H5")"
        cp "$CANONICAL_MOUNT" "$CANONICAL_H5"
        echo "  >> 已从只读挂载载入 canonical H5"
    else
        echo "  >> /artifacts/data 无 canonical H5 -> judge/full 将走 synthetic 源域 (manifest 如实标注)"
    fi
    if [ -f "$CKPT_MOUNT" ]; then
        mkdir -p checkpoints
        cp "$CKPT_MOUNT" "$CKPT"
        echo "  >> 已从只读挂载载入源域预训练 ckpt"
    fi
}

case "${1:-verify}" in
    verify)
        step "契约 Schema 快照校验"
        python -c "\
import json, glob, jsonschema; \
schemas = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob('schemas/*.schema.json'))]; \
assert len(schemas) == 8; \
[jsonschema.validators.validator_for(s) for s in schemas]; \
print('contract schemas ok:', len(schemas))"
        step "测试套件 (缺失外部数据的测试自动 skip)"
        python -m pytest tests/ -q --tb=short -p no:cacheprovider
        step "验证通过"
        echo "全部测试通过 (verify 只跑代码/fixture/Schema/导入检查, 不依赖大数据工件)"
        ;;

    reproduce_judge)
        step "reproduce_judge (评审用小规模端到端)"
        mount_artifacts
        ARGS=("${@:2}")
        OUTPUT_DIR="/outputs/judge"
        prev=""
        for a in ${ARGS[@]+"${ARGS[@]}"}; do
            [ "$prev" = "--output" ] && OUTPUT_DIR="$a"
            prev="$a"
        done
        ARGS+=(--output "$OUTPUT_DIR")
        python scripts/reproduce_judge.py ${ARGS[@]+"${ARGS[@]}"}
        echo "结果输出至 ${OUTPUT_DIR} (manifest/metrics/run.log/REPRODUCE_OK)"
        ;;

    reproduce_full)
        step "reproduce_full (完整端到端)"
        mount_artifacts
        ARGS=("${@:2}")
        OUTPUT_DIR="/outputs/full"
        prev=""
        for a in ${ARGS[@]+"${ARGS[@]}"}; do
            [ "$prev" = "--output" ] && OUTPUT_DIR="$a"
            prev="$a"
        done
        ARGS+=(--output "$OUTPUT_DIR")
        python scripts/reproduce_full.py ${ARGS[@]+"${ARGS[@]}"}
        echo "结果输出至 ${OUTPUT_DIR} (manifest/metrics/run.log/REPRODUCE_OK)"
        ;;

    reproduce)
        # 契约 v1.0 兼容入口: quick 对齐 judge, full 对齐 reproduce_full (不吞错)
        # 支持 `reproduce --mode quick --output DIR` 与 `reproduce quick DIR` 两种形态
        MODE="quick"; OUTPUT_DIR="/outputs/reproduced/phased_array"
        shift
        while [ $# -gt 0 ]; do
            case "$1" in
                --mode) MODE="$2"; shift 2 ;;
                --output) OUTPUT_DIR="$2"; shift 2 ;;
                quick|smoke|full) MODE="$1"; shift ;;
                *) OUTPUT_DIR="$1"; shift ;;
            esac
        done
        case "$MODE" in
            quick|smoke)
                step "reproduce --mode quick -> reproduce_judge"
                mount_artifacts
                python scripts/reproduce_judge.py --output "$OUTPUT_DIR/judge"
                ;;
            full)
                step "reproduce --mode full -> reproduce_full (200 轨迹, 5 seeds)"
                mount_artifacts
                python scripts/reproduce_full.py --output "$OUTPUT_DIR/full"
                ;;
            *)
                echo "Unknown mode: $MODE (expected: quick | full)"
                exit 1
                ;;
        esac
        echo "结果输出至 ${OUTPUT_DIR}"
        ;;

    *)
        exec "$@"
        ;;
esac
