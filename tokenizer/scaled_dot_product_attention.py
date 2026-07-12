from jaxtyping import Float, Bool
import math
import torch
from torch import Tensor
from .softmax import softmax
from einops import einsum

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    denominator = math.sqrt(Q.shape[-1])
    qk_scaled = torch.divide(qk, denominator)
    if mask is not None:
        qk_scaled = torch.where(mask, qk_scaled, float('-inf'))
    attention = einsum(softmax(qk_scaled, dim=-1), V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attention