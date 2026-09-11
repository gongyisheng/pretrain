# Int4 Global Scale

Test whether a per-tensor fp32 **global scale** lets int4 weight quantization keep an fp8 block-scale format without losing accuracy. Three runs at Qwen3-51M: bf16, int4 W4A16 with fp32 block scales, and int4 W4A16 with e4m3 block scales plus an fp32 global scale. Both quantized runs share blockwise 1D `(1, 16)` scaling; only the scale storage differs.

Config names read `w4a16_bs_<block scale>[_gs_<global scale>]`: `w4a16` is int4 weights with bf16 activations (matching `experiments/int4_granularity/`), `bs` the block-scale dtype, `gs` the global-scale dtype.

## Mental model

Blockwise int4 stores one scale per 16 contiguous weights. With `(1, 16)` blocks the scale metadata is a real cost: an fp32 scale adds 2 bits per weight on top of the 4-bit code, so scale storage is a third of the tensor. Dropping the scale to e4m3 cuts that to 0.5 bits per weight, but e4m3 has 4 exponent bits and 3 mantissa bits, so a raw scale can both overflow its range and round coarsely.

The global scale is what makes the narrow scale format usable. Before quantizing, the tensor is divided by

```
global_scale = amax(x) / (qmax(int4) * finfo(e4m3).max) = amax(x) / (7 * 448)
```

so every block scale lands inside e4m3's finite range, and the block scales are stored *relative* to that fp32 factor (`src/quant/quantize.py:quantize_operand`). This is the NVFP4 scale scheme applied to an int4 element: two-level scaling, fp32 per tensor and e4m3 per block.

## Hypothesis

If the global scale does its job, the e4m3 arm should match the fp32 arm's validation loss while carrying 4× less scale metadata. The residual error is e4m3's 3-bit mantissa on the block scale — roughly 6% worst-case scale error — which should be small next to int4's own 16-code resolution. A visible loss gap would mean scale precision, not scale range, is the binding constraint at `(1, 16)`.

## Setup

| Config | Weights / acts | Block shape | Scale dtype | Global scale | Scale bits/weight | Approx. params |
|---|---|---|---|---|---:|---:|
| [qwen3_51m_bf16.yaml](qwen3_51m_bf16.yaml) | bf16 / bf16 | — | — | — | — | ~51M |
| [qwen3_51m_int4_w4a16_bs_fp32.yaml](qwen3_51m_int4_w4a16_bs_fp32.yaml) | int4 / bf16 | (1, 16) | fp32 | no | 2.00 | ~51M |
| [qwen3_51m_int4_w4a16_bs_fp8_e4m3_gs_fp32.yaml](qwen3_51m_int4_w4a16_bs_fp8_e4m3_gs_fp32.yaml) | int4 / bf16 | (1, 16) | fp8_e4m3 | fp32, per tensor | 0.50 | ~51M |

Effective bits per weight: 6.00 for the fp32 arm, 4.50 for the e4m3 arm (the per-tensor fp32 factor is negligible).

All runs use 8 layers, `d_model=512`, GQA with 8 query / 4 KV heads, QK norm, `intermediate_size=1536`, OpenWebText with `tokenizers/custom_bpe_50k` and a 1% validation split. Sequence length 1024, batch size 128, gradient accumulation 2 (effective batch 256, 262,144 tokens/step), 50K steps, seed 42, bf16 mixed precision, Muon with `match_rms_adamw`, lr=5e-4, weight decay=0.1, cosine schedule (1500 warmup, min lr=5e-5), `checkpoint_every: 5000`, `eval_every: 100`, `eval_steps: 100`.

`lm_head` is excluded from quantization; embeddings, norms, attention, residuals, loss, and optimizer state stay bf16/fp32. `act` and `grad_out` are bf16 passthrough, so their `block_shape` entries are required by the config but unused — every GEMM runs the one-sided fake-quantization fallback with a bf16 operand, and only the weight carries scales.

W&B run names are `qwen3-51m-bf16` and `qwen3-51m-int4-w4a16-bs-fp32` / `qwen3-51m-int4-w4a16-bs-fp8-e4m3-gs-fp32`; checkpoints go to `checkpoints/int4_global_scale/<config_name>/`.

## Run

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_idx> nohup bash experiments/int4_global_scale/run.sh > logs/int4_global_scale_51m.log 2>&1 &
```

The script runs bf16 first, then the two int4 arms. Extra training arguments are forwarded to every run, e.g. `--no-wandb --training.early_stop=1000`.

## Results

W&B project: `pretrain-int4-global-scale`.

Report validation loss as the mean and standard deviation over the final 10 evaluations.

| Model | Recipe | Scale bits/weight | Mean Val Loss | Std. Dev. | Delta vs BF16 | Val BPB | Tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 51M | BF16 | — | TBD | TBD | 0 | TBD | TBD |
| 51M | int4 W4A16, fp32 block scale | 2.00 | TBD | TBD | TBD | TBD | TBD |
| 51M | int4 W4A16, e4m3 block scale + fp32 global | 0.50 | TBD | TBD | TBD | TBD | TBD |

## Notes

- The primary number is e4m3+global minus fp32 at fixed block shape; it isolates the scale-storage effect. The fp32-minus-bf16 gap is the reference cost of int4 weights at `(1, 16)`.
- Read `train-quant/sqnr/*` and `train-quant/underflow_rate/*` for the weight series. If the two arms' weight SQNR matches but loss diverges, the difference is not coming from weight quantization error.
- Both arms use the fake-quantization fallback, so throughput differences are quantization overhead, not GEMM speed. Neither arm demonstrates an int4 kernel speedup.
- The global scale is per tensor, computed fresh from `amax` each step. It costs one extra reduction and one broadcast divide per quantized weight.
