#!/usr/bin/env bash
# GaN RFALT -> LEO T/R -> 阵列 HI -> damage-state 实验独立复现入口。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${1:-configs/phased_array_gan.yaml}"

"${PYTHON_BIN}" -m src.sim.gan_rfalt_sim --config "${CONFIG}" --smoke \
  --output data/simulated/phased_array_gan/rfalt_smoke.h5
"${PYTHON_BIN}" -m src.data.preprocess.gan_rfalt_features --config "${CONFIG}" \
  --input data/simulated/phased_array_gan/rfalt_smoke.h5 \
  --output data/features/phased_array_gan/source/rfalt_source.h5
"${PYTHON_BIN}" -m src.sim.phased_array_sim --config "${CONFIG}" --n_traj 20
"${PYTHON_BIN}" -m src.sim.build_array_hi --config "${CONFIG}"
"${PYTHON_BIN}" -m src.experiments.run_gan_transfer --config "${CONFIG}" --smoke --n-seeds 3
