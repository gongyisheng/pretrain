# Muon vs AdamW under FP8

Does Muon's per-step advantage over AdamW survive FP8 quantization? Compare the two optimizers at Qwen3-51M with FP8 **tensorwise** GEMMs (in-house `src/quant/`, cuBLASLt via `torch._scaled_mm`), holding everything except `optimizer.name` fixed.

## Hypothesis

Muon orthogonalizes each 2D weight's momentum via Newton–Schulz, equalizing the update's singular values so every direction in a matrix gets a comparable step — unlike AdamW's per-coordinate normalization. In bf16 this gives Muon a per-step edge on the matrix-shaped params (attention/MLP projections). The question here is whether that edge holds once the GEMM operands are quantized to FP8.

**tensorwise** is the noisiest recipe (one scale per tensor, most aggressive dynamic-range compression), so it stresses each optimizer's robustness to quantization noise the hardest. Two plausible outcomes:
1. Muon keeps its edge — orthogonalization is computed on the (bf16) momentum buffer, so the update geometry is unaffected by FP8 GEMM noise; the gap tracks the bf16 result.
2. Muon's edge shrinks — FP8 noise in the forward/grad GEMMs perturbs the momentum Muon orthogonalizes, eroding the clean singular-value equalization that gives it the advantage.

bf16 references for both optimizers live in `experiments/muon_optm/` (bf16 Muon vs AdamW at 51/57M) and `experiments/fp8/` (AdamW bf16 vs FP8 recipes); this folder isolates the optimizer axis under fixed FP8 tensorwise.

## Setup

Both configs are identical except `optimizer.name` (and the Muon-only hyperparams). Muon uses the hybrid `MuonAdamWOptimizer`: 2D hidden weights → Muon, everything else (embeddings, `lm_head`, RMSNorm scales) → AdamW. `adjust_lr_fn=match_rms_adamw` matches Muon's update RMS to AdamW so it reuses the AdamW-tuned `lr`/`wd` directly.

| Config | Optimizer | Precision | Recipe | LR | min_lr | WD | Eff. batch | Approx params |
|---|---|---|---|---|---|---|---|---|
| qwen3_51m_fp8_adamw | AdamW | fp8 | tensorwise | 5e-4 | 5e-5 | 0.1 | 256 | ~51M |
| qwen3_51m_fp8_muon  | Muon  | fp8 | tensorwise | 5e-4 | 5e-5 | 0.1 | 256 | ~51M |

- Model: d_model=512, 8 layers, gqa 8/4, qk_norm, intermediate_size=1536, rope θ=10000.
- FP8 block: `dtype_recipe: fp8` (weight/act `fp8_e4m3`, grad `fp8_e5m2`), `scaling.granularity: tensorwise`, `exclude: [lm_head]`. Only eligible `nn.Linear` GEMM operands are cast to FP8 on the fly; RoPE, RMSNorm, qk_norm, attention, SwiGLU, residuals, embeddings, lm_head, cross-entropy, and optimizer state stay bf16/fp32. hp master weights preserved.
- Muon hyperparams at defaults: `momentum=0.95`, `nesterov=true`, `ns_steps=5`, shared `eps=1e-8`.
- All runs: seq_len=1024, batch=16, grad_accum=16 (eff. batch=256, ~262K tok/step), 50K steps (~13B tokens), bf16 mixed precision, cosine schedule with 1500 warmup, seed=42, OpenWebText.
- `eval_every=100`, `eval_steps=100`, `checkpoint_every=5000`, `log_optimizer_svd_metrics: true`.

Hardware requirement: SM 8.9+ (Ada/Hopper/Blackwell). On the dev box (RTX PRO 6000, SM 12.0) the FP8 GEMMs use Blackwell's cuBLASLt path.

## Run

The script runs the AdamW config followed by the Muon config.

```bash
nohup bash experiments/fp8_muon/run.sh > logs/fp8_muon_51m.log 2>&1 &
```

## Results

| Model | Optimizer | Precision | Final Val Loss | Val BPB | Δ vs AdamW |
|---|---|---|---|---|---|
| 51M | AdamW | fp8 tensorwise | TBD | TBD | — |
| 51M | Muon  | fp8 tensorwise | TBD | TBD | TBD |

## Notes

- Compare the FP8 Muon−AdamW gap here against the bf16 Muon−AdamW gap in `experiments/muon_optm/` — a preserved gap supports outcome (1), a shrunk gap supports (2).
- `optim/momentum_norm` and `optim/variance_norm` in W&B reflect only the AdamW-routed params (Muon's `momentum_buffer` is not aggregated by `metric_utils`).
- tensorwise is the most aggressive recipe; if Muon's edge collapses here, rerun with `rowwise` / `rowwise_with_gw_hp` (see `experiments/fp8/`) to see whether tighter scaling restores it.
- If turnaround matters, compare the 5K/10K-step intermediate eval losses before committing both runs to 50K.
