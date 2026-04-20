#!/bin/bash
# Sweep lm_head_lr_mult on Qwen3 57M and 0.5B (untied runs + one tied baseline per scale).
# Usage: nohup bash experiments/lm_head_lr/run.sh > logs/lm_head_lr.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(
    qwen3_57m_tied
    qwen3_57m_untied_mult1.0
    qwen3_57m_untied_mult0.5
    qwen3_57m_untied_mult0.3
    qwen3_57m_untied_mult0.2
    qwen3_57m_untied_mult0.1
    qwen3_0.5b_tied
    qwen3_0.5b_untied_mult1.0
    qwen3_0.5b_untied_mult0.5
    qwen3_0.5b_untied_mult0.3
    qwen3_0.5b_untied_mult0.2
    qwen3_0.5b_untied_mult0.1
)

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/lm_head_lr/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All lm_head_lr runs complete ==="
