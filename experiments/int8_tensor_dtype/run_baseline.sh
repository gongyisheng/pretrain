#!/bin/bash
# Run the shared bf16 baseline for the int8 tensor-dtype sweeps at 51M on Qwen3.
# Usage: nohup bash experiments/int8_tensor_dtype/run_baseline.sh > logs/int8_tensor_dtype_baseline.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

config=qwen3_51m_bf16
echo "=== ${config} ==="
echo "Started at: $(date)"
uv run python scripts/train.py --config "experiments/int8_tensor_dtype/${config}.yaml"
echo "Finished at: $(date)"

echo "=== 51M int8_tensor_dtype baseline complete ==="
