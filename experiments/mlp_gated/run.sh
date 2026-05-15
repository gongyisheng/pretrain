#!/bin/bash
# Run MLP gating ablation experiments (on vs off)
# Usage: nohup bash experiments/mlp_gated/run.sh > logs/mlp_gated.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for size in 57m 0.5b; do
    for gated in on off; do
        config="qwen3_${size}_mlp_gated_${gated}"
        echo "=== ${config} ==="
        echo "Started at: $(date)"
        uv run python scripts/train.py --config "experiments/mlp_gated/${config}.yaml"
        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=== All MLP gated runs complete ==="
