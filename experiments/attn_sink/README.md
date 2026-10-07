# Attention Sink Ablation

## Hypothesis

Adding a learned attention sink improves pretraining loss by letting each query head assign probability mass to a virtual zero-value token when no context token should be attended to.

## Setup

| Config | Attention sink | Approx. parameters | Key settings |
|---|---:|---:|---|
| [qwen3_51m_attn_sink_off.yaml](qwen3_51m_attn_sink_off.yaml) | false | 50,931,200 | Qwen3 GQA, 8 layers, d_model=512, 8 query / 4 KV heads, FlexAttention |
| [qwen3_51m_attn_sink_on.yaml](qwen3_51m_attn_sink_on.yaml) | true | 50,931,264 | Same as attention-sink-off configuration; 64 additional sink parameters |

Both runs use sequence length 1024, OpenWebText with `tokenizers/custom_bpe_50k`, batch size 16, gradient accumulation 16, bf16, MuonAdam (lr=5e-4, weight decay=0.1), cosine decay with 1,500 warmup steps and min lr=5e-5, 50,000 steps, and seed 42. Checkpoints are saved every 5,000 steps; evaluation runs every 100 steps for 100 batches.

## Run

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_idx> bash experiments/attn_sink/run.sh
```

Pass trainer overrides to both runs:

```bash
CUDA_VISIBLE_DEVICES=<free_idx> bash experiments/attn_sink/run.sh --no-wandb --training.early_stop=100
```

## Results

| Config | Final validation loss | Notes |
|---|---:|---|
| `qwen3_51m_attn_sink_off` | Pending | |
| `qwen3_51m_attn_sink_on` | Pending | |

## Notes

Each attention layer has one zero-initialized, learned scalar logit per query head. The virtual zero-value sink adds `exp(sink)` to the softmax denominator, so this model adds 8 layers × 8 heads = 64 parameters. Attention sinks require FlexAttention; both configurations set `attn_implementation: flex_attention`.

The runner executes attention-sink-off first, then attention-sink-on. W&B project: `pretrain-attn-sink`; checkpoints: `checkpoints/attn_sink/<config>/`. This is a single-seed comparison.
