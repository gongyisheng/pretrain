#!/bin/bash
# Run determinism experiment: deterministic vs non-deterministic training
# Usage: nohup bash experiments/determinism/run.sh > logs/determinism.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

CONFIGS=(
    "qwen3_57m_eff_batch_256_ga_16_determin"
    "qwen3_57m_eff_batch_256_ga_16_nondetermin"
    "qwen3_57m_eff_batch_256_ga_32_determin"
    "qwen3_57m_eff_batch_256_ga_32_nondetermin"
)

for cfg in "${CONFIGS[@]}"; do
    config="experiments/determinism/${cfg}.yaml"
    echo "=== ${cfg} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All determinism runs complete ==="
