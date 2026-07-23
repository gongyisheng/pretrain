#!/bin/bash
# moe_layer0_balance / dense-first: baseline vs. replacing layer-0 MoE with a
# compute-matched dense SwiGLU MLP.
# Usage: nohup bash experiments/moe_layer0_balance/run_dense_first.sh > logs/moe_layer0_balance_dense_first.log 2>&1 &
set -e
cd "$(dirname "$0")/../.."

configs=(
    qwen3_188m_a51m
    qwen3_171m_a51m_dense_first
)

for cfg in "${configs[@]}"; do
    echo "=== ${cfg} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_layer0_balance/${cfg}.yaml"
    echo "Finished at: $(date)"
    echo ""
done
echo "=== All dense_first runs complete ==="
