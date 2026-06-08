"""Softmax Collapse, in one F.softmax() call.

Liu et al. 2025 (arXiv:2605.06152, formalizing Prieto et al. 2024)
identify a numerical pathology in fp32 cross-entropy:

  Once the correct-class logit z_r exceeds the runner-up by more than
  ~ (mantissa_bits - 1) * ln 2,  the log-sum-exp absorbs the smaller
  exponentials.  Then  p_correct  rounds to exactly 1.0,  and the
  correct-class gradient  g_correct = p_correct - 1  rounds to exactly 0.

  fp32 threshold:  23 * ln 2  ~=  15.94
  fp64 threshold:  52 * ln 2  ~=  36.04

The other-class gradients g_k = exp(z_k - z_r) stay nonzero, so the
per-sample zero-sum identity  sum_k g_k = 0  breaks.  That residual is
what drives Numerical Feature Inflation and the Slingshot spike.

This script just calls F.softmax() with rising margins and prints what
happens before and after collapse.  No training, no plots.

Run:
    uv run python docs/grokking_wd_spike/repro_softmax_collapse.py
"""

import math

import torch
import torch.nn.functional as F


K = 97   # vocab size for the mod-97 grokking task in the paper

# Logit vector with one large `correct` class and K-1 zeros.  The margin
# is z_correct - max_{k != correct} z_k, i.e. just z_correct itself.
def logits_with_margin(margin: float) -> torch.Tensor:
    z = torch.zeros(K)
    z[0] = margin
    return z


def row(margin: float):
    z = logits_with_margin(margin)
    p32 = F.softmax(z.float(),  dim=-1)
    p64 = F.softmax(z.double(), dim=-1)
    return {
        "margin":          margin,
        "fp32_p_correct":  p32[0].item(),
        "fp32_g_correct":  (p32[0] - 1.0).item(),
        "fp32_zero_sum":   (p32[0] - 1.0 + p32[1:].sum()).item(),
        "fp64_p_correct":  p64[0].item(),
        "fp64_g_correct":  (p64[0] - 1.0).item(),
    }


print(f"fp32 SC threshold = 23 ln 2 = {23 * math.log(2):.4f}")
print(f"fp64 SC threshold = 52 ln 2 = {52 * math.log(2):.4f}\n")

print(f"{'margin':>7}  {'fp32 p_correct':>22}  {'fp32 g_correct':>15}  "
      f"{'fp32 sum_k g_k':>15}  {'fp64 g_correct':>15}")
print("-" * 80)
for m in [5.0, 10.0, 15.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0]:
    r = row(m)
    print(f"{r['margin']:7.1f}  {r['fp32_p_correct']:22.16f}  "
          f"{r['fp32_g_correct']:+15.3e}  {r['fp32_zero_sum']:+15.3e}  "
          f"{r['fp64_g_correct']:+15.3e}")

print()
print("Read across:")
print("  margin <= 15  -> fp32 still resolves a nonzero g_correct.")
print("  margin >= 20  -> fp32 p_correct == 1.0 exactly,  g_correct == 0.")
print("  fp32 sum_k g_k stays nonzero after collapse -> zero-sum broken.")
print("  fp64 g_correct survives all the way to margin 35 (= fp64 threshold).")
