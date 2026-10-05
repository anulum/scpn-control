#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Seeded CPU PPO candidate workflow.
# Plan: --dry-run [--output-dir PATH] [TIMESTEPS [EVAL_EPISODES]].
# Execution needs a recorded campaign and nine fresh candidate paths.
# Three seed-labelled legacy files do not establish independent training.
set -euo pipefail

MODE=train
OUTPUT_DIR=weights
while [[ $# -gt 0 ]]; do
  case "$1" in
  --dry-run)
    MODE=plan
    shift
    ;;
  --output-dir)
    if [[ $# -lt 2 ]]; then
      echo "An output directory is required." >&2
      exit 2
    fi
    OUTPUT_DIR=$2
    shift 2
    ;;
  --help)
    echo "Usage: train_rl_upcloud.sh [--dry-run] [--output-dir PATH] [TIMESTEPS [EVAL_EPISODES]]"
    exit 0
    ;;
  --*)
    echo "Unknown workflow option." >&2
    exit 2
    ;;
  *) break ;;
  esac
done
if [[ $# -gt 2 ]]; then
  echo "At most two counts are accepted." >&2
  exit 2
fi
TIMESTEPS=${1:-500000}
EVAL_EPISODES=${2:-50}
PYTHON_BIN=${SCPN_RL_PYTHON:-python}
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$REPO_DIR"
export CUDA_VISIBLE_DEVICES=""

if [[ "$MODE" == plan ]]; then
  for SEED in 42 123 456; do
    "$PYTHON_BIN" tools/train_rl_tokamak.py --timesteps "$TIMESTEPS" --eval-episodes "$EVAL_EPISODES" \
      --seed "$SEED" --output "$OUTPUT_DIR/ppo_tokamak_seed${SEED}.zip" --dry-run
  done
  exit 0
fi

"$PYTHON_BIN" tools/train_rl_tokamak.py --timesteps "$TIMESTEPS" --eval-episodes "$EVAL_EPISODES" \
  --seed 42 --output "$OUTPUT_DIR/ppo_tokamak_seed42.zip" --dry-run >/dev/null
"$PYTHON_BIN" tools/rl_training_results.py preflight "$OUTPUT_DIR"
mkdir -p -- "$OUTPUT_DIR"
for SEED in 42 123 456; do
  "$PYTHON_BIN" tools/train_rl_tokamak.py --timesteps "$TIMESTEPS" --eval-episodes "$EVAL_EPISODES" \
    --seed "$SEED" --output "$OUTPUT_DIR/ppo_tokamak_seed${SEED}.zip"
done
BEST_SEED=$("$PYTHON_BIN" tools/rl_training_results.py select "$OUTPUT_DIR")
cp -- "$OUTPUT_DIR/ppo_tokamak_seed${BEST_SEED}.zip" "$OUTPUT_DIR/ppo_tokamak.zip"
cp -- "$OUTPUT_DIR/ppo_tokamak_seed${BEST_SEED}.metrics.json" "$OUTPUT_DIR/ppo_tokamak.metrics.json"
"$PYTHON_BIN" benchmarks/rl_vs_classical.py --episodes "$EVAL_EPISODES" \
  --agent "$OUTPUT_DIR/ppo_tokamak.zip" --output "$OUTPUT_DIR/rl_vs_classical.json"
"$PYTHON_BIN" tools/rl_training_results.py summarize "$OUTPUT_DIR/rl_vs_classical.json"
