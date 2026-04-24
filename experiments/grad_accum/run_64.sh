#!/bin/bash
# Run all eff_batch_64 grad_accum experiments sequentially
# Usage: nohup bash experiments/grad_accum/run_64.sh > logs/grad_accum_64.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

GA_STEPS=(1 2 4 8 16 32 64)

for ga in "${GA_STEPS[@]}"; do
    config="experiments/grad_accum/qwen3_57m_eff_batch_64_ga_${ga}.yaml"
    echo "=== eff_batch=64 ga=${ga} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All eff_batch_64 runs complete ==="
