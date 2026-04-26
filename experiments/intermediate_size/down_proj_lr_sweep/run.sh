#!/bin/bash
# Run μP-lite down_proj LR scale sweep at 3 widths (mult=1, 4, 16).
# 3 widths x 5 scales = 15 runs, each early-stopped at step 12000 via debug.max_steps.
# Usage: nohup bash experiments/intermediate_size/down_proj_lr_sweep/run.sh > logs/intermediate_size_down_proj_lr_sweep.log 2>&1 &

set -e
cd "$(dirname "$0")/../../.."

for mult in 1 4 16; do
    for xt in xt0_25 xt0_5 xt1 xt2 xt4; do
        config="qwen3_57m_mult${mult}_${xt}"
        echo "=== ${config} ==="
        echo "Started at: $(date)"
        uv run python scripts/train.py --config "experiments/intermediate_size/down_proj_lr_sweep/${config}.yaml"
        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=== All down_proj LR sweep runs complete ==="
