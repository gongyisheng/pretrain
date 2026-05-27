#!/usr/bin/env bash
# Loss-function comparison sweep for grokking (sub, wd=0.1).
#
# Runs 4 configs side by side:
#   ce, ce_fp64, mse, mse_fp64
#
# Configs live in experiments/grokking/loss/.
#
# Tokenizer and sub-task data are prepared on demand via prepare.sh (idempotent).
#
# Usage:
#   # all 4 variants in parallel
#   nohup bash experiments/grokking/run_loss.sh > logs/grokking_loss.log 2>&1 &
#
#   # single variant
#   nohup bash experiments/grokking/run_loss.sh ce_fp64 > logs/grokking_loss.log 2>&1 &
#
#   # pin to a GPU
#   CUDA_VISIBLE_DEVICES=1 nohup bash experiments/grokking/run_loss.sh > logs/grokking_loss.log 2>&1 &
#
# MAX_CONCURRENCY (default 4) controls parallelism. Each run gets its own
# per-config log under logs/grokking/sub_wd0.1_<variant>.log.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

VARIANTS=(ce ce_fp64 mse mse_fp64)
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
PER_RUN_LOG_DIR="logs/grokking"

bash experiments/grokking/prepare.sh sub

mkdir -p "$PER_RUN_LOG_DIR"

run_one() {
    local variant="$1"
    local cfg="experiments/grokking/loss/qwen3_1m_sub_wd0.1_${variant}.yaml"
    local log="${PER_RUN_LOG_DIR}/sub_wd0.1_${variant}.log"
    echo "[run_loss.sh] $cfg → $log"
    uv run python scripts/train.py --config "$cfg" >"$log" 2>&1
}

if [ $# -eq 1 ]; then
    run_one "$1"
    exit 0
fi

echo "[run_loss.sh] launching loss sweep with MAX_CONCURRENCY=$MAX_CONCURRENCY"
running=0
for variant in "${VARIANTS[@]}"; do
    run_one "$variant" &
    running=$((running + 1))
    if [ "$running" -ge "$MAX_CONCURRENCY" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait
