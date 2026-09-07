# Int4 Randomized Hadamard Transform

Test whether **int4 survives a randomized Hadamard transform (RHT)** at Qwen3-51M, across two arms — **W4A16** (int4 weights, bf16 activations) and **W4A4** (int4 weights and activations) — against a shared bf16 baseline and per-arm unrotated int4 controls. Scale granularity is fixed at blockwise `(1, 32)` in every quantized run; the only variable is the Hadamard block size. `lm_head` is excluded from quantization in every quantized run. 11 runs: 1 baseline + 2 controls + 4 Hadamard block sizes × 2 arms.

RHT preconditions a GEMM operand with a block-diagonal Hadamard of size `block_size`, each block sign-flipped by a random ±1 vector drawn from the run seed, applied along the contraction dimension before quantization and inverted after dequantization. The transform is orthonormal, so it is mathematically an identity on the GEMM; it changes only *what the quantizer sees*. This is the QuaRot / SpinQuant / QuIP# mechanism: mixing spreads outlier energy across `block_size` coordinates, pushing each scale block's distribution toward Gaussian so int4's 16 codes cover more of the mass instead of being stretched by one large element.

## Hypothesis

int4 has 16 codes, and `experiments/int4_granularity/` showed that what limits them is the dynamic range inside a scale block. Shrinking the block is the brute-force fix and it costs stored bits. RHT attacks the same problem for free: it flattens the distribution rather than paying for more scales.

**Headline test.** Unrotated int4 at `(1, 32)` costs 5.00 effective bits/weight; unrotated int4 at `(1, 16)` costs 6.00. If RHT at `(1, 32)` matches or beats the unrotated `(1, 16)` cell from `experiments/int4_granularity/`, then a rotation buys finer-granularity accuracy at coarser-granularity cost, and int4 becomes a basis problem rather than a scale-budget problem.

- **W4A16.** RHT reaches only the *weight*: activations are bf16, hence never quantized and never rotated. Weights have far milder outliers than activations, so expect a modest but real gain — enough to close part of the `(1, 32)` → `(1, 16)` gap, not a rescue, since weight-only int4 at `(1, 32)` is already the healthier of the two arms.
- **W4A4.** Activations carry the outliers that 16 codes cannot absorb, and this is where RHT should pay off most. Expect the largest Δ vs control here, and it is the arm that answers "does int4 survive": if unrotated W4A4 is unstable at `(1, 32)` and a rotation makes it train, that is the result.
- **Hadamard block vs scale block.** `block_size` 32 matches the `(1, 32)` scale block exactly — one Hadamard block per scale block, the QuaRot-style pairing, where each scale block sees a fully independent mixture. `block_size` 16 mixes within *half* a scale block, so each scale still spans two independently-mixed halves and one of them can retain an outlier; expect 16 to underperform 32. `block_size` 64 and 128 mix *across* 2 and 4 scale blocks, spreading outlier energy wider but leaving neighbouring scale blocks correlated. Expect the gain to arrive by 32 and then flatten, with 64/128 buying little beyond it.
- **Cost.** The transform runs in float32 on every quantized operand every step. W4A16 pays it on weights only; W4A4 pays it on weights and activations. Storage cost is zero — the sign vector is `block_size` floats and the Hadamard is implicit — which is the entire point.

If W4A4 at `block_size` 32 lands near its W4A16 twin, int4 activations are an outlier problem that a change of basis solves.

## Setup

**Hadamard axis** (identical in both arms). Effective bits/weight counts fp32 scale storage against the weight tensor only and is `4 + 32/32 = 5.00` for every quantized run here: RHT adds no stored bits, so the whole axis sits at one storage point. int4 sets a 4-bit code range (`qmax=7`) but the tensor container is int8, so this column is a logical accounting, not measured memory.

| Rotation | `block_size` | Hadamard block vs `(1, 32)` scale block | Effective bits/weight |
|---|---|---|---|
| none (control) | — | — | 5.00 |
| hadamard | 16 | half a scale block | 5.00 |
| hadamard | 32 | exactly one scale block | 5.00 |
| hadamard | 64 | spans 2 scale blocks | 5.00 |
| hadamard | 128 | spans 4 scale blocks | 5.00 |

`random_sign: true` throughout; the sign vector is seeded from the run seed (42), so it is fixed within a run and identical across runs at equal `block_size`.

**Arm axis.** Rotation is applied per *quantized* operand — a bf16 operand is never rotated — so which GEMMs the transform actually reaches differs by arm:

| Arm | `dtype.weight` | `dtype.act` | `dtype.grad_out` | fwd GEMM | dgrad GEMM | wgrad GEMM |
|---|---|---|---|---|---|---|
| W4A16 | int4 | bf16 | bf16 | weight rotated + quantized, dequantized back to bf16 matmul (simulated) | weight rotated, simulated | no quantized operand — rotation is a no-op |
| W4A4 | int4 | int4 | bf16 | both operands rotated; fused `gemm.int8_scaled_mm` consumes the codes directly, rotation cancels across operands | weight rotated, simulated | act rotated, simulated |

`gemms: [fwd, dgrad, wgrad]` is set in every rotated run, so the rotation is declared uniformly and the arms differ only by dtype.

Config names are `qwen3_51m_int4_<arm>_hadamard_<block_size>`, with controls `qwen3_51m_int4_<arm>`.

All runs: ~51M params (`d_model=512`, 8 layers, 8/4 Q/KV heads, `intermediate_size=1536`), seq_len=1024, effective batch=256 (bf16 and unrotated controls use batch=16/grad_accum=16; Hadamard runs use batch=64/grad_accum=4), 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1500 warmup steps, min_lr=5e-5, OpenWebText, bf16 mixed precision, seed 42, `eval_every=100`, `eval_steps=100`, `checkpoint_every=5000`.

Every `block_size` must be a power of two dividing each contraction extent. The extents here are 256 (k/v_proj dgrad), 512 (all `d_model` contractions), 1536 (down_proj fwd, gate/up_proj dgrad), and the per-microbatch token count (wgrad: 16384 for controls, 65536 for Hadamard runs), so all four block sizes are legal; `lm_head` (50257) is excluded from quantization.

## Run

```bash
nohup bash experiments/int4_hadamard/run.sh > logs/int4_hadamard.log 2>&1 &
```

## Results

W&B project: `pretrain-int4-hadamard`. Baseline: `qwen3_51m_bf16`.

### W4A16 (int4 weights, bf16 activations)

| Rotation | `block_size` | Val Loss | Δ vs bf16 | Δ vs control | Val BPB | Weight rel. err | s/step |
|---|---|---|---|---|---|---|---|
| bf16 | — | | 0 | — | | — | |
| none | — | | | 0 | | | |
| hadamard | 16 | | | | | | |
| hadamard | 32 | | | | | | |
| hadamard | 64 | | | | | | |
| hadamard | 128 | | | | | | |

### W4A4 (int4 weights and activations)

| Rotation | `block_size` | Val Loss | Δ vs bf16 | Δ vs control | Δ vs W4A16 twin | Val BPB | Weight rel. err | s/step |
|---|---|---|---|---|---|---|---|---|
| none | — | | | 0 | | | | |
| hadamard | 16 | | | | | | | |
| hadamard | 32 | | | | | | | |
| hadamard | 64 | | | | | | | |
| hadamard | 128 | | | | | | | |

`Δ vs control` is the effect of the rotation at that `block_size`. `Δ vs W4A16 twin` isolates the residual cost of int4 activations once both arms are rotated the same way.

### Headline: does RHT buy a granularity step?

Unrotated `(1, 16)` numbers come from `experiments/int4_granularity/qwen3_51m_int4_<arm>_blockwise1d_16`, which shares every hyperparameter with this experiment.

| Arm | unrotated (1, 32) — 5.00 bits | best RHT (1, 32) — 5.00 bits | unrotated (1, 16) — 6.00 bits | RHT closes the gap? |
|---|---|---|---|---|
| W4A16 | | | | |
| W4A4 | | | | |

## Notes

- Compare each int4 run with this experiment's bf16 baseline using the mean validation loss over the final 10 evaluations.
- The two unrotated controls duplicate `experiments/int4_granularity/qwen3_51m_int4_<arm>_blockwise1d_32` exactly — same granularity, same hyperparameters, same seed. They are re-run here so this experiment is self-contained; they should reproduce those cells, so a mismatch is a determinism bug worth chasing rather than a result.
- `weight rel. err` is measured in the *unrotated* basis: `accumulate_quantization_sums` squares `source - dequantized`, and `dequantize_operand` inverts the rotation first. So it is directly comparable across rotated and control runs and is the cleanest single-number read on whether the transform helped the quantizer. Compare its trend with validation loss to separate quantization error from training dynamics.
- **Underflow is the metric RHT should move most**, and it is measured in the *rotated* basis — `record_operand` passes `rotated_source` so the nonzero mask asks "was a value the quantizer actually saw flushed to code 0". A scale set by one outlier is exactly what flushes the rest of a block to zero, so if RHT is working, underflow should drop before validation loss does. Track it per arm alongside `weight rel. err`.
- Record instability, divergence, and loss spikes, not just final-window loss. Unrotated W4A4 at `(1, 32)` is the cell most likely to diverge; if it does and a rotated twin does not, note the step it diverged instead of reporting only a final number.
- **Step time is comparable across `block_size` within an arm** — granularity is fixed, so fusion coverage is identical across a column and the only delta is the transform itself, which makes this the experiment that prices RHT. **Do not compare step time across arms:** fused `gemm.int8_scaled_mm` covers only GEMMs whose *both* operands are int4 (forward only in W4A4, none in W4A16), and everything else dequantizes into a bf16 matmul, so the arms are ordered by how much fusion they get rather than by quantization cost.
- int4 shares the int8 GEMM path because int4 values are stored in an int8 container, so W4A4 buys no memory or bandwidth over int8 here. This experiment measures the *accuracy* effect of a change of basis on a 4-bit code range, not packed-int4 throughput.
- The granularity curve this experiment sits on is `experiments/int4_granularity/`; the same arms are run there across six granularities with identical hyperparameters, so the `(1, 32)` column there is the reference this experiment perturbs.
