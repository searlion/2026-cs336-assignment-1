import torch
from torch import Tensor
from jaxtyping import Float
from einops import rearrange

class Multihead_self_attention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_model d_model"],
        k_proj_weight: Float[Tensor, " d_model d_model"],
        v_proj_weight: Float[Tensor, " d_model d_model"],
        o_proj_weight: Float[Tensor, " d_model d_model"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj_weight_sliced = rearrange(q_proj_weight, "d_model (num_heads d_k) -> d_model num_heads d_k", num_heads=num_heads)
        self.k_proj_weight_sliced = rearrange(k_proj_weight, "d_model (num_heads d_k) -> d_model num_heads d_k", num_heads=num_heads)
        self.v_proj_weight_sliced = rearrange(v_proj_weight, "d_model (num_heads d_v) -> d_model num_heads d_v", num_heads=num_heads)
        self.o_prj_weight = o_proj_weight

    def forward(
        self, 
        in_features: Float[Tensor, " ... sequence_length d_model"]
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        