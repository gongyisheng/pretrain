#!/bin/bash
# Fixed-tokens batch size sweep: ~13.1B tokens per run (8 * 32 * 1024 * 50000).
# Each bs has its own max_steps so the total token budget stays constant.
# Usage: nohup bash experiments/batch_size/run_tokens_13b.sh > logs/batch_size_tokens_13b.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

token=13b
for bs in 16k 64k 256k 1m; do
    config="qwen3_57m_bs_${bs}_token_${token}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/batch_size/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All fixed-tokens runs complete ==="
