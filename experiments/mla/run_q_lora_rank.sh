#!/bin/bash
# Phase 5 — q_lora_rank sweep (nope=32, rope=32, v_head=64, kv_lora=512 fixed): query factored through a q_lora_rank latent.
# Usage: nohup bash experiments/mla/run_q_lora_rank.sh > logs/mla_q_lora_rank.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=()
for q in 64 128 256 384 512; do
    configs+=("qwen3_57m_mla_qlora_${q}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/mla/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Phase 5 (q_lora_rank) complete ==="
