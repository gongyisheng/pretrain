#!/bin/bash
# Full pretraining pipeline: preprocess data + train GPT-2
# Run with: nohup bash scripts/run_pipeline.sh > pipeline.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Preprocessing data ==="
echo "Started at: $(date)"

# Clean partial files from any previous interrupted run
rm -f data/all_tokens.bin

# Only preprocess if train.bin doesn't exist
if [ ! -f data/train.bin ]; then
    python scripts/preprocess_data.py --config configs/gpt2_small.yaml
    echo "Preprocessing complete at: $(date)"
else
    echo "data/train.bin already exists, skipping preprocessing"
fi

echo ""
echo "=== Step 2: Training GPT-2 ==="
echo "Started at: $(date)"
python scripts/train.py --config configs/gpt2_small.yaml

echo ""
echo "=== Pipeline complete ==="
echo "Finished at: $(date)"
