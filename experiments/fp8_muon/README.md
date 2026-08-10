# Muon vs AdamW under FP8

Does Muon's per-step advantage over AdamW survive FP8 quantization, and does that depend on the grad format? Sweep two optimizers × two FP8 dtype assignments at Qwen3-51M with FP8 **tensorwise** GEMMs (in-house `src/quant/`, cuBLASLt via `torch._scaled_mm`), holding everything else fixed.

The two dtype assignments differ only in the gradient GEMM operands:
- **std** — weight/act `fp8_e4m3`, grad (`grad_input`/`grad_weight`) `fp8_e5m2`. The standard mixed-format recipe: e5m2 trades mantissa for exponent range on the wider-dynamic-range gradients.
- **alle4m3** — every operand (weight/act/grad) `fp8_e4m3`. Keeps grad mantissa precision at the cost of dynamic range; stresses whether the extra range in e5m2 actually matters here.

Both are written as explicit `dtype` maps (no named `dtype_recipe`) so the grad-format axis is visible in the config.

## Hypothesis

Muon orthogonalizes each 2D weight's momentum via Newton–Schulz, equalizing the update's singular values so every direction in a matrix gets a comparable step — unlike AdamW's per-coordinate normalization. In bf16 this gives Muon a per-step edge on the matrix-shaped params (attention/MLP projections). The question here is whether that edge holds once the GEMM operands are quantized to FP8.

**tensorwise** is the noisiest recipe (one scale per tensor, most aggressive dynamic-range compression), so it stresses each optimizer's robustness to quantization noise the hardest. Two plausible outcomes:
1. Muon keeps its edge — orthogonalization is computed on the (bf16) momentum buffer, so the update geometry is unaffected by FP8 GEMM noise; the gap tracks the bf16 result.
2. Muon's edge shrinks — FP8 noise in the forward/grad GEMMs perturbs the momentum Muon orthogonalizes, eroding the clean singular-value equalization that gives it the advantage.

## Setup

All four configs are identical except `optimizer.name` (+ Muon-only hyperparams) and the grad `dtype` (std vs alle4m3). Muon uses the hybrid `MuonAdamWOptimizer`: 2D hidden weights → Muon, everything else (embeddings, `lm_head`, RMSNorm scales) → AdamW. `adjust_lr_fn=match_rms_adamw` matches Muon's update RMS to AdamW so it reuses the AdamW-tuned `lr`/`wd` directly.

| Config | Optimizer | Precision | Grad fmt | LR | min_lr | WD | Eff. batch | Approx params |
|---|---|---|---|---|---|---|---|---|
| qwen3_51m_bf16_adamw        | AdamW | bf16           | —     | 5e-4 | 5e-5 | 0.1 | 256 | ~51M |
| qwen3_51m_bf16_muon         | Muon  | bf16           | —     | 5e-4 | 5e-5 | 0.1 | 256 | ~51M |
| qwen3_51m_fp8_std_adamw     | AdamW | fp8 tensorwise | e5m2  | 5e-4 | 5e-5 | 0.1 | 256 | ~51M |
| qwen3_51m_fp8_std_muon      | Muon  | fp8 tensorwise | e5m2  | 5e-4 | 5e-5 | 0.1 | 256 | ~51M |
| qwen3_51m_fp8_alle4m3_adamw | AdamW | fp8 tensorwise | e4m3  | 5e-4 | 5e-5 | 0.1 | 256 | ~51M |
| qwen3_51m_fp8_alle4m3_muon  | Muon  | fp8 tensorwise | e4m3  | 5e-4 | 5e-5 | 0.1 | 256 | ~51M |

The bf16 pair is the full-precision reference (`quant.enabled: false`): it anchors the bf16 Muon−AdamW gap this experiment asks about, so the FP8 gaps can be read as degradation relative to it rather than against the external `experiments/muon_optm/` run.

- Model: d_model=512, 8 layers, gqa 8/4, qk_norm, intermediate_size=1536, rope θ=10000.
- FP8 block: explicit `dtype` map (std: weight/act `fp8_e4m3` + grad `fp8_e5m2`; alle4m3: all operands `fp8_e4m3`), `scaling.granularity: tensorwise`, `exclude: [lm_head]`. Only eligible `nn.Linear` GEMM operands are cast to FP8 on the fly; RoPE, RMSNorm, qk_norm, attention, SwiGLU, residuals, embeddings, lm_head, cross-entropy, and optimizer state stay bf16/fp32. hp master weights preserved. (cuBLAS rejects e5m2×e5m2, so e4m3 weight/act is required either way.)
- Muon hyperparams at defaults: `momentum=0.95`, `nesterov=true`, `ns_steps=5`, shared `eps=1e-8`.
- All runs: seq_len=1024, batch=16, grad_accum=16 (eff. batch=256, ~262K tok/step), 50K steps (~13B tokens), bf16 mixed precision, cosine schedule with 1500 warmup, seed=42, OpenWebText.
- `eval_every=100`, `eval_steps=100`, `checkpoint_every=5000`, `log_layer_weight_svd_metrics: true`.

Hardware requirement: SM 8.9+ (Ada/Hopper/Blackwell). On the dev box (RTX PRO 6000, SM 12.0) the FP8 GEMMs use Blackwell's cuBLASLt path.

## Run

The script runs the bf16 baselines first, then sweeps `{std, alle4m3} × {adamw, muon}`.

```bash
nohup bash experiments/fp8_muon/run.sh > logs/fp8_muon_51m.log 2>&1 &
```

## Results

| Model | Optimizer | Precision / Grad fmt | Final Val Loss | Val BPB | Δ vs AdamW |
|---|---|---|---|---|---|
| 51M | AdamW | bf16           | TBD | TBD | — |
| 51M | Muon  | bf16           | TBD | TBD | TBD |
| 51M | AdamW | fp8 e5m2 (std)     | TBD | TBD | — |
| 51M | Muon  | fp8 e5m2 (std)     | TBD | TBD | TBD |
| 51M | AdamW | fp8 e4m3 (alle4m3) | TBD | TBD | — |
| 51M | Muon  | fp8 e4m3 (alle4m3) | TBD | TBD | TBD |

## Notes

- Compare the FP8 Muon−AdamW gap here against the in-experiment bf16 baseline pair — a preserved gap supports outcome (1), a shrunk gap supports (2). (`experiments/muon_optm/` is a looser external reference.)
- `optim/momentum_norm` and `optim/variance_norm` in W&B reflect only the AdamW-routed params (Muon's `momentum_buffer` is not aggregated by `metric_utils`).
- tensorwise is the most aggressive recipe; if Muon's edge collapses here, rerun with `rowwise` / `rowwise_with_gw_hp` (see `experiments/fp8_granularity/`) to see whether tighter scaling restores it.
- If turnaround matters, compare the 5K/10K-step intermediate eval losses before committing all runs to 50K.
- std vs alle4m3 isolates the grad format: if alle4m3 degrades (or diverges), e5m2's extra dynamic range on gradients is load-bearing at tensorwise scaling; if it matches std, the grad mantissa precision wins the trade. Watch whether the effect differs between AdamW and Muon.
