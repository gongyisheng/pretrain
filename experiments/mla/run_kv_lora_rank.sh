#!/bin/bash
# Phase 2 — kv_lora_rank sweep (nope=32, rope=32, v_head=64 fixed), scaling the KV latent down from the center (512).
# Usage: nohup bash experiments/mla/run_kv_lora_rank.sh > logs/mla_kv_lora_rank.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=()
for kv in 64 128 256 384; do
    configs+=("qwen3_57m_mla_kvlora_${kv}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/mla/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Phase 2 (kv_lora_rank) complete ==="
