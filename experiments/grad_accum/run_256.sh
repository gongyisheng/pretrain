#!/bin/bash
# Run all eff_batch_256 grad_accum experiments sequentially
# Usage: nohup bash experiments/grad_accum/run_256.sh > logs/grad_accum_256.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

GA_STEPS=(8 16 32 64 128 256)

for ga in "${GA_STEPS[@]}"; do
    config="experiments/grad_accum/qwen3_57m_eff_batch_256_ga_${ga}.yaml"
    echo "=== eff_batch=256 ga=${ga} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "${config}"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All eff_batch_256 runs complete ==="
