#!/bin/bash
# moe_layer0_balance / l0dense: baseline vs. replacing layer-0 MoE with a
# compute-matched dense SwiGLU MLP.
# Usage: nohup bash experiments/moe_layer0_balance/run_l0dense.sh > logs/moe_layer0_balance_l0dense.log 2>&1 &
set -e
cd "$(dirname "$0")/../.."

configs=(
    qwen3_188m_a51m
    qwen3_171m_a51m_l0dense
)

for cfg in "${configs[@]}"; do
    echo "=== ${cfg} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_layer0_balance/${cfg}.yaml"
    echo "Finished at: $(date)"
    echo ""
done
echo "=== All l0dense runs complete ==="
