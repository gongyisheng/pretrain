#!/bin/bash
# MLA experiments on Qwen3 57M: baseline (MLA vs GQA vs MHA) + three sweeps
# (kv_lora_rank, q_lora_rank, qk_rope_head_dim). Each variant is its own
# standalone YAML — no CLI overrides. The MLA center (kv_lora=256, q_lora=0,
# qk_rope=32) is qwen3_57m_mla.yaml and is the shared row in every sweep table.
# Usage: nohup bash experiments/mla/run.sh > logs/mla.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(
    qwen3_57m_mha
    qwen3_57m_gqa
    qwen3_57m_mla
    qwen3_57m_mla_kvlora128
    qwen3_57m_mla_kvlora384
    qwen3_57m_mla_kvlora512
    qwen3_57m_mla_qlora384
    qwen3_57m_mla_rope16
    qwen3_57m_mla_rope48
    qwen3_57m_mla_rope64
)

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/mla/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All MLA runs complete ==="
