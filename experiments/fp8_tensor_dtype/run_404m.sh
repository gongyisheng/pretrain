#!/bin/bash
# Run FP8 tensor-dtype cells at Qwen3 404M (e4m3 forward subset).
# Usage (single GPU, sequential):
#   nohup bash experiments/fp8_tensor_dtype/run_404m.sh > logs/fp8_tensor_dtype_404m.log 2>&1 &
#
# 4-GPU parallel (run nvidia-smi first; one config per GPU on GPUs 0-3):
#   nohup env CUDA_VISIBLE_DEVICES=0 bash -c 'cd "'"$PWD"'"; c=qwen3_404m_fp8_e4m3_w8a16;  echo "=== $c $(date)"; uv run python scripts/train.py --config experiments/fp8_tensor_dtype/$c.yaml' > logs/fp8_tensor_dtype_404m_gpu0.log 2>&1 &
#   nohup env CUDA_VISIBLE_DEVICES=1 bash -c 'cd "'"$PWD"'"; c=qwen3_404m_fp8_e4m3_w16a8;  echo "=== $c $(date)"; uv run python scripts/train.py --config experiments/fp8_tensor_dtype/$c.yaml' > logs/fp8_tensor_dtype_404m_gpu1.log 2>&1 &
#   nohup env CUDA_VISIBLE_DEVICES=2 bash -c 'cd "'"$PWD"'"; c=qwen3_404m_fp8_e4m3_w8a8;   echo "=== $c $(date)"; uv run python scripts/train.py --config experiments/fp8_tensor_dtype/$c.yaml' > logs/fp8_tensor_dtype_404m_gpu2.log 2>&1 &
#   nohup env CUDA_VISIBLE_DEVICES=3 bash -c 'cd "'"$PWD"'"; c=qwen3_404m_fp8_e4m3_w8a8g8; echo "=== $c $(date)"; uv run python scripts/train.py --config experiments/fp8_tensor_dtype/$c.yaml' > logs/fp8_tensor_dtype_404m_gpu3.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_404m_bf16)

for tensor_dtypes in w8a16 w16a8 w8a8 w8a8g8 w8a8_e5m2_g8; do
    configs+=("qwen3_404m_fp8_e4m3_${tensor_dtypes}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_tensor_dtype/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 404M fp8_tensor_dtype runs complete ==="
