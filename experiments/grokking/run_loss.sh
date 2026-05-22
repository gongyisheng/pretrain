#!/usr/bin/env bash
# Loss-function comparison sweep for grokking (sub, wd=0.1).
#
# Runs 4 configs side by side:
#   ce, ce_fp64, mse, mse_fp64
#
# Reuses the tokenizer and sub-task data prepared by run.sh; if either is
# missing, run experiments/grokking/run.sh first to build them.
#
# Usage:
#   # all 4 variants in parallel
#   nohup bash experiments/grokking/run_loss.sh > logs/grokking/loss_sweep.log 2>&1 &
#
#   # single variant
#   bash experiments/grokking/run_loss.sh ce_fp64
#
#   # pin to a GPU
#   CUDA_VISIBLE_DEVICES=1 nohup bash experiments/grokking/run_loss.sh > logs/grokking/loss_sweep.log 2>&1 &
#
# MAX_CONCURRENCY (default 4) controls parallelism. Each run gets its own
# per-config log under logs/grokking/sub_wd0.1_<variant>.log.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

VARIANTS=(ce ce_fp64 mse mse_fp64)
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
PER_RUN_LOG_DIR="logs/grokking"
TOKENIZER_FILE="tokenizers/grokking/tokenizer.json"
DATA_DIR="data/grokking_sub_p97_f0.3"

if [ ! -f "$TOKENIZER_FILE" ]; then
    echo "[run_loss.sh] tokenizer missing: $TOKENIZER_FILE"
    echo "  → run experiments/grokking/run.sh first to build tokenizer + data"
    exit 1
fi

if [ ! -f "${DATA_DIR}/train.bin" ] || [ ! -f "${DATA_DIR}/val.bin" ]; then
    echo "[run_loss.sh] tokenized sub data missing under $DATA_DIR"
    echo "  → run experiments/grokking/run.sh first to build data"
    exit 1
fi

mkdir -p "$PER_RUN_LOG_DIR"

run_one() {
    local variant="$1"
    local cfg="experiments/grokking/qwen3_1m_sub_wd0.1_${variant}.yaml"
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
