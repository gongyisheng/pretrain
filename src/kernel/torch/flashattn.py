import torch
import torch.nn.functional as F


@torch.compile(dynamic=True)
def torch_flash_attn(q, k, v, causal=True, attn_mask=None, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    if attn_mask is not None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=sm_scale)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)
