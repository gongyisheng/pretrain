# Attention Residuals

Check whether Block AttnRes improves loss at each model scale, and how the improvement shifts the scaling law curve.

Reference: [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)

## Technique

Standard transformers accumulate layer outputs with fixed-weight residual connections:
```
x = x + sublayer(norm(x))
```

Block AttnRes replaces this with a learned, softmax-normalized attention over all preceding block outputs:
```
h = softmax_over_blocks(w_l · norm(blocks)) @ blocks
x = x + sublayer(norm(h))
```

Applied twice per layer (before self-attention and before MLP). Layers are grouped into blocks; standard residuals accumulate within a block, and attention aggregates across block boundaries.

## Models

Fixed width (d=512, 8 heads) across all runs — only depth varies. Two architectures for comparison.

### GPT-2 (MHA + LayerNorm + GELU FFN)

| Config | d_model | layers | heads | ~Params | block_size | blocks | LR |
|---|---|---|---|---|---|---|---|
| gpt2_d512_l4 | 512 | 4 | 8 | ~39M | 2 | 2 | 1e-3 |
| gpt2_d512_l8 | 512 | 8 | 8 | ~51M | 2 | 4 | 8e-4 |
| gpt2_d512_l12 | 512 | 12 | 8 | ~64M | 2 | 6 | 8e-4 |
| gpt2_d512_l16 | 512 | 16 | 8 | ~77M | 2 | 8 | 6e-4 |
| gpt2_d512_l20 | 512 | 20 | 8 | ~89M | 2 | 10 | 6e-4 |

### Qwen3 (GQA + RoPE + RMSNorm + SwiGLU)

| Config | d_model | layers | heads | kv_heads | ~Params | block_size | blocks | LR |
|---|---|---|---|---|---|---|---|---|
| qwen3_d512_l4 | 512 | 4 | 8 | 4 | ~41M | 2 | 2 | 1e-3 |
| qwen3_d512_l8 | 512 | 8 | 8 | 4 | ~57M | 2 | 4 | 8e-4 |
| qwen3_d512_l12 | 512 | 12 | 8 | 4 | ~73M | 2 | 6 | 8e-4 |
| qwen3_d512_l16 | 512 | 16 | 8 | 4 | ~89M | 2 | 8 | 6e-4 |
| qwen3_d512_l20 | 512 | 20 | 8 | 4 | ~104M | 2 | 10 | 6e-4 |

`block_size` = number of full transformer layers per block. Block boundaries fire when `layer_idx % block_size == 0`.

Extra parameters per layer: 2 × `Linear(d_model, 1)` + 2 × `RMSNorm(d_model)` ≈ 4 × d_model — negligible (<0.03% of total).

## Run

```bash
# Single run
python scripts/train.py --config experiments/attn_res/qwen3_d512_l4.yaml

# All sizes sequentially
bash experiments/attn_res/run.sh
```

## Debug

```bash
# Resume from a checkpoint, run a few steps, capture spikes with a new W&B run
python scripts/train.py --config experiments/attn_res/qwen3_d512_l12.yaml \
    --resume checkpoints/attn_res/qwen3_d512_l12/step_25000.pt \
    --logging.wandb_run_name=debug-spike-qwen3-d512-l12 \
    --debug.max_steps=29999 \
    --debug.spike.enabled=true \
    --debug.spike.grad_norm_threshold=5.0

# Inspect per-weight stats of a checkpoint
python scripts/debug_weight_stats.py --ckpt checkpoints/attn_res/qwen3_d512_l4/step_10000.pt

# Inspect optimizer state (grad_norm, noise, snr, eff_step) of a checkpoint
python scripts/debug_optim.py --ckpt checkpoints/attn_res/qwen3_d512_l4/step_10000.pt
```

## W&B

Project: `pretrain-attn-res`. Compare against `pretrain-scaling-law` runs of the same size.

Custom chart: X=`train/flops` (log scale), Y=`train/loss` (log scale), grouped by run.

## Results

TODO
