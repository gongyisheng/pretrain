#!/bin/bash
# MoE warmup-steps sweep, sigmoid router, aux-loss 1e-3.
# Model: qwen3_183m_a51m -> 64 routed, top-8, 0 shared.
# Usage: nohup bash experiments/moe_warmup/run_s0r8.sh > logs/moe_warmup_s0r8.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

warmups=(500 1000 1500 2000)

configs=()
for w in "${warmups[@]}"; do
    configs+=("qwen3_183m_a51m_warmup${w}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_warmup/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All moe_warmup s0r8 runs complete ==="
