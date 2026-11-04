# GQA number of heads

## Hypothesis

At fixed model width and GQA ratio, changing the number of query heads changes how attention partitions its representations. This sweep measures the effect on validation quality, throughput, and memory relative to the eight-query-head Qwen3 51M baseline.

## Setup

All configurations copy `configs/qwen3_51m.yaml`, changing only the query/KV head counts, checkpoint directory, and W&B identifiers. The GQA ratio is fixed at two query heads per KV head, with `n_kv_heads = n_heads / 2` and `head_dim = 512 / n_heads`.

| Config | Query heads | KV heads | Head dimension | Parameters (approximately 50.93M) |
| --- | ---: | ---: | ---: | ---: |
| [qwen3_51m_h2.yaml](qwen3_51m_h2.yaml) | 2 | 1 | 256 | 50,934,288 |
| [qwen3_51m_h4.yaml](qwen3_51m_h4.yaml) | 4 | 2 | 128 | 50,932,256 |
| [qwen3_51m_h8.yaml](qwen3_51m_h8.yaml) | 8 | 4 | 64 | 50,931,264 |
| [qwen3_51m_h16.yaml](qwen3_51m_h16.yaml) | 16 | 8 | 32 | 50,930,816 |
| [qwen3_51m_h32.yaml](qwen3_51m_h32.yaml) | 32 | 16 | 16 | 50,930,688 |

Fixed settings:

- Model: width 512, eight layers, vocabulary 50,257, dense MLP width 1,536, RMSNorm, QK normalization, attention sinks, and RoPE with theta 10,000; all dropout is zero.
- Data: OpenWebText, custom 50K BPE tokenizer, sequence length 1,024, validation split 0.01, four data workers, and seed 42.
- Training: batch size 16, gradient accumulation 16 (effective batch 256 sequences), 50,000 steps, bf16, and gradient clipping 1.0.
- Optimizer: MuonAdam, learning rate `5e-4`, weight decay 0.1; AdamW branch betas `(0.9, 0.99)` and epsilon `1e-8`; Muon branch momentum 0.95, Nesterov enabled, `match_rms_adamw` adjustment, and epsilon `1e-8`.
- Schedule: cosine decay, 1,500 warmup steps, and minimum learning rate `5e-5`.
- Cadence: checkpoint every 5,000 steps, evaluate every 100 steps over 100 batches, and log every 10 steps.

## Run

From the repository root, inspect GPU usage and select a free device:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free-device> bash experiments/gqa_n_heads/run.sh
```

Replace `<free-device>` with the available GPU index before running. The runner trains all five configurations sequentially in ascending head-count order and forwards additional arguments to `scripts/train.py`.

Checkpoints are isolated under `checkpoints/gqa_n_heads/qwen3_51m_h<N>/`. W&B uses project `pretrain-gqa-n-heads` and run names `qwen3-51m-h<N>`.

## Results

Training has not been launched; all results are pending.

| Query heads | Validation loss | Validation BPB | Throughput (tokens/s) | Peak GPU memory (GiB) |
| ---: | --- | --- | --- | --- |
| 2 | Pending | Pending | Pending | Pending |
| 4 | Pending | Pending | Pending | Pending |
| 8 (baseline) | Pending | Pending | Pending | Pending |
| 16 | Pending | Pending | Pending | Pending |
| 32 | Pending | Pending | Pending | Pending |

## Notes

- Compare each run with `h8` at the same training step and on the same GPU for performance measurements.
- Q/K/V projection widths stay fixed: query width 512 and key/value width 256 each. The head dimension and number of attention score matrices change with the head count.
- Only QK RMSNorm and attention-sink parameters cause the tiny parameter-count variation. Per layer, QK RMSNorm contributes `2 * head_dim` parameters and sinks contribute `n_heads`.
