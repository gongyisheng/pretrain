#!/bin/bash
# Select a free GPU with nvidia-smi, then set CUDA_VISIBLE_DEVICES before running.
# Usage: nohup bash experiments/fp8_module_sensitivity/run.sh > logs/fp8_module_sensitivity.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_77m_bf16)
for recipe in w8a16 w8a8 w8a8g8; do
    for module in attn mlp lm_head; do
        configs+=("qwen3_77m_fp8_${recipe}_${module}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_module_sensitivity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== fp8_module_sensitivity runs complete ==="
