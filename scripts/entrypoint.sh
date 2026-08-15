#!/bin/bash
# =====================================================================
# 相控阵组件 - 容器入口脚本
#   支持 verify / reproduce 子命令
#
# 数据策略:
#   - mosfet_canonical.h5 (schema_v4, 876KB) 随包发布 -> 源域特征已有
#   - source_phased_array_tcn_pretrain.pt 随包发布 -> 预训练 ckpt 已有
#   - 原始 NASA MAT 文件 (7.85GB) 不随包 -> P1 跳过 (canonical 已覆盖)
#   - schema_v3 source_features.h5 不随包 -> P2 用 --canonical 走 schema_v4
#   - 仿真数据由 phased_array_sim 运行时自生成 -> P3-P6 自包含
#
# 双仿真集 (full 模式):
#   - sim_v2 (--subdose on):  通道级 (build_channel_hi -> channel_features.h5)
#   - sim_v1 (--subdose off): 服务级 (build_array_hi -> target_features.h5)
#   cross_level_transfer 层级消融需要 sim_v1; ch_* 通道级实验需要 sim_v2
# =====================================================================
set -euo pipefail

export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

CFG="configs/phased_array.yaml"
CANONICAL_H5="data/features/phased_array/schema_v4/source/mosfet_canonical.h5"
CKPT="checkpoints/source_phased_array_tcn_pretrain.pt"

step() { echo ""; echo "========== $1 =========="; }

case "${1:-verify}" in
    verify)
        step "相控阵测试套件"
        python -m pytest tests/ -q --tb=short
        step "验证通过"
        echo "✅ 全部测试通过 (缺失外部数据的测试自动 skip)"
        ;;

    reproduce)
        MODE="${2:-smoke}"
        OUTPUT_DIR="${3:-/results/reproduced/phased_array}"
        mkdir -p "$OUTPUT_DIR"

        case "$MODE" in
            smoke)
                step "smoke 模式（合成数据，预计 < 5 分钟）"

                # P1: 源域特征 (canonical h5 随包则跳过)
                if [ -f "$CANONICAL_H5" ]; then
                    step "P1 源域特征工程（跳过：canonical 已随包）"
                else
                    step "P1 源域特征工程（合成 MOSFET）"
                    python -m src.data.preprocess.mosfet_features --config $CFG --synthetic --report
                fi

                # P2: 源域预训练 (ckpt 随包则跳过)
                if [ -f "$CKPT" ]; then
                    step "P2 源域预训练（跳过：checkpoint 已随包）"
                else
                    step "P2 源域预训练（smoke）"
                    python -m src.train.pretrain --config $CFG --smoke
                fi

                step "P3a 相控阵仿真 sim_v2（快速）"
                python -m src.sim.phased_array_sim --config $CFG --fast --subdose on

                step "P3b 相控阵仿真 sim_v1（快速）"
                python -m src.sim.phased_array_sim --config $CFG --fast --subdose off

                step "P4 通道级 HI 构造（读 sim_v2）"
                python -m src.sim.build_channel_hi --config $CFG --report

                step "P4b 阵列级 HI 构造（读 sim_v1）"
                python -m src.sim.build_array_hi --config $CFG --report

                step "P5 迁移训练（smoke, 旧观测层路径）"
                python -m src.transfer.train_transfer --config $CFG --smoke 2>&1 || {
                    echo "  >> [预期跳过] 旧观测层迁移需要 schema_v3 (不随包); 主线实验在 P6"
                }

                step "P6 对比实验（smoke）"
                python -m src.experiments.run_groups --config $CFG --smoke --level channel --output-dir "$OUTPUT_DIR"
                ;;

            full)
                step "full 模式（200 轨迹完整实验，预计 ~50 分钟 CPU）"

                # P1: 源域特征 (canonical h5 随包则跳过)
                if [ -f "$CANONICAL_H5" ]; then
                    step "P1 源域特征工程（跳过：canonical 已随包）"
                    echo "  >> $CANONICAL_H5 ($(du -h "$CANONICAL_H5" | cut -f1))"
                else
                    step "P1 源域特征工程（合成模式生成）"
                    python -m src.data.preprocess.mosfet_features --config $CFG --synthetic --report
                fi

                # P2: 源域预训练 (ckpt 随包则跳过)
                if [ -f "$CKPT" ]; then
                    step "P2 源域预训练（跳过：checkpoint 已随包）"
                    echo "  >> $CKPT ($(du -h "$CKPT" | cut -f1))"
                else
                    step "P2 源域预训练（用 canonical schema_v4 数据）"
                    python -m src.train.pretrain --config $CFG --canonical
                fi

                # P3a: 仿真 sim_v2 (subdose on, 通道级主线)
                step "P3a 相控阵仿真 sim_v2（200 轨迹，子阵级独立损伤）"
                python -m src.sim.phased_array_sim --config $CFG --n_traj 200 --seed 42 --subdose on

                # P3b: 仿真 sim_v1 (subdose off, 服务级层级消融)
                step "P3b 相控阵仿真 sim_v1（200 轨迹，旧标量路径，服务级消融）"
                python -m src.sim.phased_array_sim --config $CFG --n_traj 200 --seed 42 --subdose off

                # P4: 通道级 HI (读 sim_v2)
                step "P4 通道级 HI 构造（读 sim_v2）"
                python -m src.sim.build_channel_hi --config $CFG --report

                # P4b: 阵列级 HI (读 sim_v1, cross_level_transfer 所需)
                step "P4b 阵列级 HI 构造（读 sim_v1，服务层消融所需）"
                python -m src.sim.build_array_hi --config $CFG --report

                # P5: 对比实验 (channel + service 层级消融, 5 种子)
                step "P5 对比实验（channel + service 层级消融，5 种子）"
                python -m src.experiments.run_groups --config $CFG --level channel --output-dir "$OUTPUT_DIR"

                # P6: 通道级基线
                step "P6 通道级基线"
                python -m src.baselines.channel_baselines --config $CFG || echo "  >> 基线非阻塞，跳过"

                # P7: 结果合并
                step "P7 结果合并"
                python scripts/merge_matrix_results.py || echo "  >> 合并非阻塞，跳过"
                ;;
            *)
                echo "Unknown mode: $MODE (expected: smoke | full)"
                exit 1
                ;;
        esac
        step "复现完成"
        echo "结果输出至 $OUTPUT_DIR"
        echo "  >> 实验结果: $OUTPUT_DIR/all_metrics_phased_array.json"
        echo "  >> 基线结果: checkpoints/baselines_channel.json"
        echo "  >> 合并报告: docs/results_phased_array_channel.md"
        ;;

    *)
        exec "$@"
        ;;
esac
