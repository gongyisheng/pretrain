#!/usr/bin/env bash
# AdamW eps ablation sweep for grokking (sub, wd=0.1): eps ∈ {1e-15,1e-12,1e-8,1e-7,1e-6,1e-5} × loss ∈ {ce, ce_fp64}.
# Usage: nohup bash experiments/grokking/run_adamw_eps.sh > logs/grokking_adamw_eps.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

EPS_VALUES=(1e-15 1e-12 1e-8 1e-7 1e-6 1e-5)
LOSSES=(ce ce_fp64)

VARIANTS=()
for loss in "${LOSSES[@]}"; do
    for eps in "${EPS_VALUES[@]}"; do
        VARIANTS+=("${loss}_eps${eps}")
    done
done

MAX_CONCURRENCY="${MAX_CONCURRENCY:-6}"
PER_RUN_LOG_DIR="logs/grokking"

bash experiments/grokking/prepare.sh sub

mkdir -p "$PER_RUN_LOG_DIR"

run_one() {
    local variant="$1"
    local cfg="experiments/grokking/adamw_eps/qwen3_1m_sub_wd0.1_${variant}.yaml"
    local log="${PER_RUN_LOG_DIR}/sub_wd0.1_${variant}.log"
    if [ ! -f "$cfg" ]; then
        echo "[run_adamw_eps.sh] config missing: $cfg"
        return 1
    fi
    echo "[run_adamw_eps.sh] $cfg → $log"
    uv run python scripts/train.py --config "$cfg" >"$log" 2>&1
}

if [ $# -eq 1 ]; then
    run_one "$1"
    exit 0
fi

echo "[run_adamw_eps.sh] launching eps sweep (${#VARIANTS[@]} variants) with MAX_CONCURRENCY=$MAX_CONCURRENCY"
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
