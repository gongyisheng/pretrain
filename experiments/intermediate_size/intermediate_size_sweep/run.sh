#!/bin/bash
# Sweep FFN expansion ratio (intermediate_size / d_model) at fixed lr=5e-4.
# 9 widths (mult 0.25 - 32), each early-stopped at step 12000 via debug.max_steps.
# Usage: nohup bash experiments/intermediate_size/intermediate_size_sweep/run.sh > logs/intermediate_size_sweep.log 2>&1 &

set -e
cd "$(dirname "$0")/../../.."

for mult in 0_25 0_5 1 2 3 4 8 16 32; do
    config="qwen3_57m_mult${mult}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/intermediate_size/intermediate_size_sweep/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All intermediate-size sweep runs complete ==="
