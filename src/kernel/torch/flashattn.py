import torch
import torch.nn.functional as F


def torch_flash_attn(q, k, v, causal=True, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)
