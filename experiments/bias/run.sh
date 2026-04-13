#!/bin/bash
# Run bias ablation experiments (Qwen3)
# Usage: nohup bash experiments/bias/run.sh > logs/bias.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for variant in none all attn mlp lm_head; do
    config="qwen3_57m_bias_${variant}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/bias/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All bias ablation runs complete ==="
