#!/bin/bash
# Run all intermediate_size LR sweep experiments sequentially (smallest mult first).
# 8 widths x 4 LRs = 32 runs, each early-stopped at step 12000 via debug.max_steps.
# Usage: nohup bash experiments/intermediate_size/lr_sweep/run.sh > logs/intermediate_size_lr_sweep.log 2>&1 &

set -e
cd "$(dirname "$0")/../../.."

for mult in 1 2 3 4 6 8 12 16; do
    for lr in 1e-4 2e-4 5e-4 1e-3; do
        config="qwen3_57m_mult${mult}_lr${lr}"
        echo "=== ${config} ==="
        echo "Started at: $(date)"
        uv run python scripts/train.py --config "experiments/intermediate_size/lr_sweep/${config}.yaml"
        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=== All intermediate_size LR sweep runs complete ==="
