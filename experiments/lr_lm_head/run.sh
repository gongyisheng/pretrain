#!/bin/bash
# Sweep lr_mult.lm_head on Qwen3 57M and 0.5B (untied runs + one tied baseline per scale).
# Usage: nohup bash experiments/lr_lm_head/run.sh > logs/lr_lm_head.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

sizes=(57m 0.5b)
mults=(1.0 0.5 0.3 0.2 0.1)

run() {
    local config="$1"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/lr_lm_head/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
}

for size in "${sizes[@]}"; do
    run "qwen3_${size}_tied"
    for mult in "${mults[@]}"; do
        run "qwen3_${size}_untied_mult${mult}"
    done
done

echo "=== All lr_lm_head runs complete ==="
