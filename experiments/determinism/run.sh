#!/bin/bash
# Run determinism experiment: deterministic vs non-deterministic training
# Usage: nohup bash experiments/determinism/run.sh > logs/determinism.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for mode in deterministic nondeterministic; do
    config="experiments/determinism/qwen3_57m_${mode}.yaml"
    echo "=== ${mode} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All determinism runs complete ==="
