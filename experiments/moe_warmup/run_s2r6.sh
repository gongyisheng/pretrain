#!/bin/bash
# MoE warmup-steps sweep, sigmoid router, aux-loss 1e-3.
# Model: qwen3_188m_a51m_s2r6 -> 64 routed, top-6, 2 shared (8 active experts).
# Usage: nohup bash experiments/moe_warmup/run_s2r6.sh > logs/moe_warmup_s2r6.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

warmups=(500 1000 1500 2000)

configs=()
for w in "${warmups[@]}"; do
    configs+=("qwen3_188m_a51m_s2r6_warmup${w}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_warmup/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All moe_warmup s2r6 runs complete ==="
