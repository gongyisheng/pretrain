#!/bin/bash
# Run the 200,000-step optimizer comparison.

set -euo pipefail
cd "$(dirname "$0")/../.."

configs=()
optimizers=(muonc muonw muonh)
lrs=(1e-2)
for optimizer in "${optimizers[@]}"; do
    for lr in "${lrs[@]}"; do
        configs+=("qwen3_51m_${optimizer}_lr${lr}_steps_200k")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/muonh/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done
