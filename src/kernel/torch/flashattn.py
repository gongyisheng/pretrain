import torch
import torch.nn.functional as F


@torch.compile
def torch_flash_attn(q, k, v, attn_mask=None, is_causal=False, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=sm_scale)
