#!/bin/bash
# Batch-size sweep on qwen3_133m_a45m (top-8): fix batch_size=8, sweep gradient
# accumulation (effective batch). max_steps scaled inversely so every run sees
# the same ~13.1B-token budget. Run smallest effective batch first.
# Usage: nohup bash experiments/moe_batch_size/run_a45m.sh > logs/moe_batch_size_a45m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

GRAD_ACCUM=(32 64 128 256)

for ga in "${GRAD_ACCUM[@]}"; do
    config="experiments/moe_batch_size/qwen3_133m_a45m_ga${ga}.yaml"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All batch-size runs complete ==="
