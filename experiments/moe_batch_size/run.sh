#!/bin/bash
# Batch-size sweep on qwen3_183m_a51m (top-8): fix batch_size=8, sweep gradient
# accumulation (effective batch). max_steps scaled inversely so every run sees
# the same ~13.1B-token budget. Run smallest effective batch first.
# Usage: nohup bash experiments/moe_batch_size/run.sh > logs/moe_batch_size.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

# effective batch size in sequences (= batch_size 8 × grad_accu)
BATCH_SIZE=(256 512 1024 2048)

for bs in "${BATCH_SIZE[@]}"; do
    config="experiments/moe_batch_size/qwen3_183m_a51m_bs${bs}.yaml"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All batch-size runs complete ==="
