#!/bin/bash
# Run int8..int4 tensor-dtype cells (W8A16, W8A8; tensorwise) vs bf16 at 51M on Qwen3.
# Usage: nohup bash experiments/int8_tensor_dtype/run_tensorwise.sh > logs/int8_tensor_dtype_tensorwise.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

# bf16 baseline is granularity-independent; it runs in run_baseline.sh.
configs=()
for bits in 8 7 6 5 4; do
    for layout in w8a16 w8a8; do
        configs+=("tensorwise/qwen3_51m_int${bits}_${layout}_tensorwise")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int8_tensor_dtype/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int8_tensor_dtype tensorwise runs complete ==="
