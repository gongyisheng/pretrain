# Per-Tensor FP8 Sensitivity (weight / act / grad_input / grad_weight)

Which of the four quantizable tensors can afford FP8, and in which element format. One tensor is quantized per cell with the other three at bf16, so each loss gap is that tensor's main effect; element format (`e4m3` / `e5m2`) is the second axis, and scaling is fixed at `tensorwise`. The bf16 run anchors the loss reference.

**Cell notation.** `w`/`a`/`dx`/`dw` are the four quantizable tensors — weight, activation, `grad_input` (dgrad's incoming gradient), `grad_weight` (wgrad's) — each followed by its width: `8` is FP8 in the row's format, `16` is bf16, and an omitted token means 16. So `w8a16` is weight-only FP8, `w16a16dw8` is grad_weight-only, `w8a8` is the weight+act pair with both gradients bf16.

Those four names are **tensors, not GEMM operands**: `weight` is quantized in the forward and in dgrad, `act` in the forward and in wgrad, `grad_input` in dgrad only, `grad_weight` in wgrad only (`src/quant/linear.py:68-119`). So a cell measures the cost of rounding one tensor everywhere it is consumed — no config key can quantize a tensor in one GEMM and leave it exact in the other. The per-GEMM axis is a fusion/throughput question and stays in `experiments/fp8_dtype/`, which owns the whole-network recipes and the backward 2×2. Design: `docs/superpowers/specs/2026-08-17-quant-operand-and-int-bit-allocation-design.md`.

## Hypothesis

Each tensor has its own distribution and its own path to the loss, and at tensorwise granularity one scale per tensor is set by its largest element — so the wide, heavy-tailed tensors waste levels.

1. **weight is cheapest** — stable and roughly symmetric, so a single tensor-wide scale covers it.
2. **act costs more than weight** — activation outliers are what one scale cannot cover.
3. **grad_input is the most sensitive of the two gradients** — its error is re-quantized and re-propagated at every layer, so it compounds with depth.
4. **grad_weight is the most tolerant** — wgrad reduces over the whole token batch (M = batch × seq), so per-element rounding noise averages down before it reaches the update.
5. **e4m3 wins on weight/act, e5m2 on the gradients** — forward tensors want mantissa since the scale already supplies range; gradient distributions are wide enough that range is binding. The format gap should be widest on whichever tensor stage 1 ranks most sensitive.
6. **Additivity** — the four single-tensor gaps against `fp8_dtype/qwen3_51m_fp8_alle4m3` (all four at e4m3). Summing to the joint gap means the errors are independent; a smaller joint gap means they partially cancel inside the same dot product.

Note the asymmetry when reading the ranking: `weight` and `act` are each consumed by two GEMMs, the two gradient tensors by one. A larger `weight` gap than `grad_input` gap is partly two exposures against one.

## Setup

All hyperparameters matched; only `training.quantization.dtype` varies. Every FP8 run uses `scaling: {granularity: tensorwise}`, `exclude: [lm_head]`, seed 42, same data order and LR schedule.

| Config | weight | act | grad_input | grad_weight | Stage | Approx params |
|---|---|---|---|---|---|---|
| qwen3_51m_bf16 | — | — | — | — | 1 | ~51M |
| qwen3_51m_fp8_e4m3_w8a16 | e4m3 | bf16 | bf16 | bf16 | 1 | ~51M |
| qwen3_51m_fp8_e4m3_w16a8 | bf16 | e4m3 | bf16 | bf16 | 1 | ~51M |
| qwen3_51m_fp8_e4m3_w16a16dx8 | bf16 | bf16 | e4m3 | bf16 | 1 | ~51M |
| qwen3_51m_fp8_e4m3_w16a16dw8 | bf16 | bf16 | bf16 | e4m3 | 1 | ~51M |
| qwen3_51m_fp8_e4m3_w8a8 | e4m3 | e4m3 | bf16 | bf16 | 1 | ~51M |
| qwen3_51m_fp8_e5m2_w8a16 | e5m2 | bf16 | bf16 | bf16 | 2 | ~51M |
| qwen3_51m_fp8_e5m2_w16a8 | bf16 | e5m2 | bf16 | bf16 | 2 | ~51M |
| qwen3_51m_fp8_e5m2_w16a16dx8 | bf16 | bf16 | e5m2 | bf16 | 2 | ~51M |
| qwen3_51m_fp8_e5m2_w16a16dw8 | bf16 | bf16 | bf16 | e5m2 | 2 | ~51M |

All runs: seq_len=1024, batch=16, grad_accum=16 (effective batch=256, ~262K tok/step), 50K steps (~13B tokens), Muon (`muon_adjust_lr_fn: match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine with 1500 warmup, min_lr=5e-5, OpenWebText, bf16 mixed precision, `eval_steps: 100` (matching the other quant experiments, not the repo default of 25).

**Readout and noise floor.** The primary number is the **mean val loss over the final 10 evals** (~steps 49.1K–50K), not the single last eval — same runs, same cost, less eval noise in the statistic that every conclusion here rests on. Read it off W&B; no code change. This base config is byte-identical to `fp8_dtype/qwen3_51m_bf16.yaml` apart from `checkpoint_dir`/`wandb_project`, so the resolution comes from that folder's three bf16 replicas (seeds 42/43/44) reduced the same way: a `Δ vs bf16` counts as real only when `|Δ| > 2σ` of those three. If that baseline has already run, drop `qwen3_51m_bf16` from `run.sh` and reuse the number.

## What runs in FP8

`scaled_mm_op` (`src/quant/utils.py`) returns a fused scaled GEMM only when **both** operands of that GEMM are quantized. Every single-tensor cell leaves one operand bf16 in each GEMM it touches, so **no cell here fuses except `w8a8`**, which fuses the forward (1 of 3 GEMMs) and still fake-quants the weight in dgrad and the act in wgrad. The unfused path quantizes, dequantizes back to the compute dtype, and runs a bf16 matmul (`src/quant/linear.py:47`); that dequantize rounds `scale × code` to bf16, a rounding the fused kernel avoids by applying scales in its fp32 epilogue. So single-tensor gaps are slight **upper** bounds on the true fp8 cost — worth remembering in the additivity check, where the joint corner (`alle4m3`) is fully fused.

Consequently **tokens/sec is not a result in this folder** — every cell emulates at least two of the three GEMMs, so all of them are numerics probes running slower than bf16. Speed belongs to `fp8_dtype`, where cells are grouped by fusion class. Loss is comparable across all ten rows.

Everything outside the quantized `nn.Linear` modules stays bf16/fp32 (embeddings, lm_head, RoPE, RMSNorm, qk_norm, attention, SwiGLU, residuals, cross-entropy, optimizer state); hp master weights preserved. Both attention and MLP linears are quantized in every cell — `apply_quantization` (`src/quant/convert.py`) swaps every eligible `nn.Linear`, so there is no module axis here. Hardware requirement for the fused path: SM 8.9+.

## Run

```bash
nohup bash experiments/fp8_tensor_dtype/run.sh 1 > logs/fp8_tensor_dtype_s1.log 2>&1 &   # e4m3 ladder
nohup bash experiments/fp8_tensor_dtype/run.sh 2 > logs/fp8_tensor_dtype_s2.log 2>&1 &   # e5m2 ladder
```

## Results

W&B project: `pretrain-fp8-tensor-dtype`. Val loss is the mean of the final 10 evals; `Δ vs bf16` is against the seed-42 baseline; `> noise?` is `|Δ| > 2σ`. `rel. quant err` is mean relative quantization error on that tensor's operand slots from `src/metrics/quant.py` (`fwd_weight`+`dgrad_weight` for weight, `fwd_act`+`wgrad_act` for act, `dgrad_grad` for grad_input, `wgrad_grad` for grad_weight).

**Stage 1 — e4m3**

| Model | cell | fp8 tensor | GEMMs touched | Val Loss | Δ vs bf16 | > noise? | Val BPB | rel. quant err |
|---|---|---|---|---|---|---|---|---|
| 51M | bf16 | — | — | | 0 | — | | — |
| 51M | w8a16 | weight | fwd, dgrad | | | | | |
| 51M | w16a8 | act | fwd, wgrad | | | | | |
| 51M | w16a16dx8 | grad_input | dgrad | | | | | |
| 51M | w16a16dw8 | grad_weight | wgrad | | | | | |
| 51M | w8a8 | weight + act | fwd, dgrad, wgrad | | | | | |
| 51M | w8a8dx8dw8 † | all four | all | | | | | |

† `fp8_dtype/qwen3_51m_fp8_alle4m3`, same base config — copy the number rather than rerunning.

**Stage 2 — e5m2**

| Model | cell | fp8 tensor | GEMMs touched | Val Loss | Δ vs bf16 | > noise? | Val BPB | rel. quant err |
|---|---|---|---|---|---|---|---|---|
| 51M | w8a16 | weight | fwd, dgrad | | | | | |
| 51M | w16a8 | act | fwd, wgrad | | | | | |
| 51M | w16a16dx8 | grad_input | dgrad | | | | | |
| 51M | w16a16dw8 | grad_weight | wgrad | | | | | |
| 51M | w8a8dx8dw8 † | all four | all | | | | | |

† `fp8_dtype/qwen3_51m_fp8_alle5m2`.

## Notes

- Per format, check additivity: the four single-tensor gaps summed vs the all-four gap. Equal means the per-tensor errors are independent; a smaller joint gap means they cancel. `w8a8` gives the same check for the weight+act pair alone.
- The `rel. quant err` column is the mechanism check. A tensor that costs more loss should show correspondingly larger error; differing losses at matching errors means the gap is not coming from quantization error at all.
- Format ranking is read within a tensor, across stages — `e4m3 weight` vs `e5m2 weight`. Comparing across tensors within a stage is the sensitivity ranking; both are clean because the format is held fixed inside each stage.
- Renamed from `fp8_op_dtype`, and the cell names kept but re-read: `w`/`a` used to mean the forward GEMM's two operands, which this config cannot isolate — the `weight` key also quantizes dgrad's weight and the `act` key also quantizes wgrad's act. They now name tensors, and `dx`/`dw` complete the set. Per-GEMM isolation would need a GEMM-scoped dtype key; per-GEMM *fusion* is already covered in `fp8_dtype`.
- `e4m3_w8a8` is also the fourth corner of `fp8_dtype`'s backward 2×2. Same base config, so run it once and share the number.
- Companions sharing this base config and noise floor: `fp8_dtype/` (recipes, backward GEMMs, throughput), `fp8_granularity/` (scale granularity), `int_dtype/` (int8→int4 ladder), `int_granularity/`.
