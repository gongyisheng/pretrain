# Int4 Stochastic Rounding

Test whether stochastic rounding of int4 weights beats round-to-nearest-even at Qwen3-51M. Three runs: bf16, int4 W4A16 with RNE, and int4 W4A16 with SR. Both quantized arms share blockwise 1D `(1, 16)` scaling with fp32 block scales; only `rounding.weight` differs.

Config names read `w4a16_<rounding>`: `w4a16` is int4 weights with bf16 activations (matching `experiments/int4_granularity/` and `experiments/int4_global_scale/`), `rne`/`sr` the weight rounding mode.

## Mental model

RNE picks the nearest of int4's 16 codes, which minimizes per-draw error but is deterministic and biased: a weight always lands on the same side of the same threshold, so its reconstruction error is a fixed function of the weight rather than noise. SR rounds down to `floor(u)` with probability `1 - frac(u)` and up otherwise, so `E[q] = u` exactly. It trades a larger per-draw error for zero bias (`src/quant/quantize.py:_compute_codes`).

Two things make the trade plausible here rather than merely theoretical:

- **The weight is re-quantized every micro-batch.** With `gradient_accumulation_steps: 16`, one optimizer step draws 16 independent roundings of the same master weight, so the accumulated gradient is taken against something much closer to the unrounded weight than any single draw is. RNE gets no such averaging — all 16 micro-batches see the identical biased reconstruction.
- **RNE weights pile up on the decision boundary.** Measured on the `experiments/int4_global_scale/` checkpoints, 27% of rounding residuals `|u - round(u)|` sit in the top 5% bin next to 0.5, against 9.4% for a uniform residual. This is the standard STE oscillation: a weight parked at a threshold flips code every time it jitters, and RNE locks each flip in for a whole step. SR turns that latch into a coin flip whose mean is correct.

## Hypothesis

Per-draw weight SQNR at `(1, 16)` measures 21.39 dB for RNE and 18.38 dB for SR on a Gaussian probe — SR is 3 dB noisier in any single forward pass. Averaged over 32 draws SR reaches 33.4 dB with a 10x smaller mean bias, while RNE does not improve at all.

So the two arms separate exactly on whether the int4 penalty is bias-dominated or variance-dominated. If bias dominates, SR should recover part of the ~0.057 nat gap that `experiments/int4_global_scale/` measured between int4 W4A16 and bf16, despite being noisier per step. If variance dominates, SR should be strictly worse and the 3 dB is simply paid. My read is a modest SR win or a wash: the residual pile-up is severe enough to be a real bias source, but weight-side SR is a weaker lever than gradient-side SR, and 16 micro-batch draws is a small averaging window.

A secondary signal: SR should *lower* the logged per-step `train-quant/sqnr/weight/*` even if validation loss improves. Those two moving in opposite directions is the result that confirms the mechanism.

## Setup

| Config | Weights / acts | Block shape | Scale dtype | Weight rounding | Approx. params |
|---|---|---|---|---|---:|
| [qwen3_51m_bf16.yaml](qwen3_51m_bf16.yaml) | bf16 / bf16 | — | — | — | ~51M |
| [qwen3_51m_int4_w4a16_rne.yaml](qwen3_51m_int4_w4a16_rne.yaml) | int4 / bf16 | (1, 16) | fp32 | RNE | ~51M |
| [qwen3_51m_int4_w4a16_sr.yaml](qwen3_51m_int4_w4a16_sr.yaml) | int4 / bf16 | (1, 16) | fp32 | SR | ~51M |

Both quantized arms carry 6.00 effective bits per weight (4-bit code plus a 2-bit amortized fp32 block scale), so this axis is free: SR costs no extra storage and no extra scale metadata.

All runs use 8 layers, `d_model=512`, GQA with 8 query / 4 KV heads, QK norm, `intermediate_size=1536`, OpenWebText with `tokenizers/custom_bpe_50k` and a 1% validation split. Sequence length 1024, batch size 16, gradient accumulation 16 (effective batch 256, 262,144 tokens/step), 50K steps, seed 42, bf16 mixed precision, Muon with `match_rms_adamw`, lr=5e-4, weight decay=0.1, cosine schedule (1500 warmup, min lr=5e-5), `checkpoint_every: 5000`, `eval_every: 100`, `eval_steps: 100`.

`lm_head` is excluded from quantization; embeddings, norms, attention, residuals, loss, and optimizer state stay bf16/fp32. `act` and `grad_out` are bf16 passthrough, so their `block_shape` entries are required by the config but unused, and their `rounding` entries default to RNE and never apply. Every GEMM runs the one-sided fake-quantization fallback with a bf16 operand, and only the weight carries scales and a rounding mode.

W&B run names are `qwen3-51m-bf16`, `qwen3-51m-int4-w4a16-rne`, and `qwen3-51m-int4-w4a16-sr`; checkpoints go to `checkpoints/int4_stochastic_rounding/<config_name>/`.

## Run

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_idx> nohup bash experiments/int4_stochastic_rounding/run.sh > logs/int4_stochastic_rounding_51m.log 2>&1 &
```

The script runs bf16 first, then RNE, then SR. Extra training arguments are forwarded to every run, e.g. `--no-wandb --training.early_stop=1000`.

## Results

W&B project: `pretrain-int4-stochastic-rounding`.

Report validation loss as the mean and standard deviation over the final 10 evaluations.

| Model | Recipe | Weight rounding | Mean Val Loss | Std. Dev. | Delta vs BF16 | Val BPB | Tokens/s |
|---|---|---|---:|---:|---:|---:|---:|
| 51M | BF16 | — | TBD | TBD | 0 | TBD | TBD |
| 51M | int4 W4A16, (1, 16) | RNE | TBD | TBD | TBD | TBD | TBD |
| 51M | int4 W4A16, (1, 16) | SR | TBD | TBD | TBD | TBD | TBD |

## Notes

- The primary number is SR minus RNE at fixed block shape and scale dtype; it isolates the rounding rule. The RNE-minus-BF16 gap is the reference cost of int4 weights at `(1, 16)`.
- Read `train-quant/sqnr/weight/*` alongside validation loss. SR is expected to lose ~3 dB there by construction, so a lower SQNR paired with an equal or lower validation loss is the intended outcome, not a contradiction.
- Also read `train-quant/underflow_rate/weight/*`. SR lets a weight below half a code still reach code 1 sometimes instead of always flushing to zero, so the underflow rate should drop even though per-draw error rises.
- SR draws fresh randomness inside the quantizer on every forward, so these runs are not bit-reproducible across restarts even at a fixed seed. Resume-and-compare checks that hold for the RNE arm will not hold for the SR arm.
- Both arms use the fake-quantization fallback, so throughput differences are quantization overhead, not GEMM speed. SR adds one `rand_like` and a compare per quantized weight element per forward; expect it to show up as a small tokens/s cost.
- The `experiments/int4_global_scale/` runs used `batch_size: 128` with `gradient_accumulation_steps: 2`. Effective batch matches at 256, but the micro-batch split does not, and the SR mechanism here depends on the number of accumulation steps. Compare the RNE arm here to the SR arm here, not across experiments.
