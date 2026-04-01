#!/usr/bin/env bash

echo "=== Run 1: baseline (no doc_mask) ==="
uv run python scripts/train.py --config experiments/cross_doc/qwen3_57m_baseline.yaml

echo "=== Run 2: doc_mask enabled ==="
uv run python scripts/train.py --config experiments/cross_doc/qwen3_57m_cross_doc.yaml

echo "=== Done. Compare runs in W&B under group: cross-doc-sweep ==="
