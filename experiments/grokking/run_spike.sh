#!/usr/bin/env bash
# Slingshot-spike ablation for grokking (sub task): tests what eliminates the spikes in the wd>0 regime.
# Usage: nohup bash experiments/grokking/run_spike.sh > logs/grokking_spike.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

VARIANTS=(wd0.0_ce wd0.0_ce_fp64 wd0.1_ce_fp64 wd0.1_ce_fp64_lion wd0.3_ce_fp64_lion wd0.1_ce_fp64_ls1e-5 wd0.1_ce_ls1e-5)
MAX_CONCURRENCY="${MAX_CONCURRENCY:-5}"
PER_RUN_LOG_DIR="logs/grokking"

bash experiments/grokking/prepare.sh sub

mkdir -p "$PER_RUN_LOG_DIR"

run_one() {
    local variant="$1"
    local cfg="experiments/grokking/spike/qwen3_1m_sub_${variant}.yaml"
    local log="${PER_RUN_LOG_DIR}/spike_${variant}.log"
    echo "[run_spike.sh] $cfg → $log"
    uv run python scripts/train.py --config "$cfg" >"$log" 2>&1
}

if [ $# -eq 1 ]; then
    run_one "$1"
    exit 0
fi

echo "[run_spike.sh] launching spike ablation with MAX_CONCURRENCY=$MAX_CONCURRENCY"
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
