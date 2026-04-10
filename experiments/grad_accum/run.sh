#!/bin/bash
# Run all grad_accum experiments sequentially (fewest accumulation steps first)
# Usage: nohup bash experiments/grad_accum/run.sh > logs/grad_accum.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

GA_STEPS=(1 2 4 8 16 32)

for ga in "${GA_STEPS[@]}"; do
    config="experiments/grad_accum/qwen3_57m_ga_${ga}.yaml"
    echo "=== ga=${ga} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All grad_accum runs complete ==="
