#!/bin/bash
# Run the int4 randomized-Hadamard sweep (4 Hadamard block sizes x W4A16/W4A4, plus
# unrotated controls) vs bf16 at 51M on Qwen3. Scale granularity is fixed at (1, 16).
# Usage: nohup bash experiments/int4_hadamard/run.sh > logs/int4_hadamard.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16)
for arm in w4a16 w4a4; do
    configs+=("qwen3_51m_int4_${arm}")
    for block in 16 32 64 128; do
        configs+=("qwen3_51m_int4_${arm}_hadamard_${block}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_hadamard/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int4_hadamard runs complete ==="
