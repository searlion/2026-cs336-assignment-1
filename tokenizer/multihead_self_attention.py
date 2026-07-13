import torch
from torch import Tensor
from jaxtyping import Float
from einops import rearrange, einsum
from tokenizer.scaled_dot_product_attention import scaled_dot_product_attention

class Multihead_self_attention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_out d_in"],
        k_proj_weight: Float[Tensor, " d_out d_in"],
        v_proj_weight: Float[Tensor, " d_out d_in"],
        o_proj_weight: Float[Tensor, " d_out d_in"],
    ):
        super().__init__()
        assert (d_model % num_heads) == 0, f"d_model is not divisible by num_heads, d_model={d_model}, num_heads={num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj_weight_sliced = torch.nn.Parameter(rearrange(q_proj_weight, "(num_heads d_k) d_in  -> num_heads d_k d_in", num_heads=num_heads))
        self.k_proj_weight_sliced = torch.nn.Parameter(rearrange(k_proj_weight, "(num_heads d_k) d_in  -> num_heads d_k d_in", num_heads=num_heads))
        self.v_proj_weight_sliced = torch.nn.Parameter(rearrange(v_proj_weight, "(num_heads d_v) d_in  -> num_heads d_v d_in ", num_heads=num_heads))
        self.o_proj_weight = torch.nn.Parameter(o_proj_weight)

    def forward(
        self, 
        in_features: Float[Tensor, " ... sequence_length d_model"]
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        q_transformed = einsum(self.q_proj_weight_sliced, in_features, "num_heads d_k d_in, ... sequence_length d_in -> ... num_heads sequence_length d_k")
        k_transformed = einsum(self.k_proj_weight_sliced, in_features, "num_heads d_k d_in, ... sequence_length d_in -> ... num_heads sequence_length d_k")
        v_transformed = einsum(self.v_proj_weight_sliced, in_features, "num_heads d_v d_in, ... sequence_length d_in -> ... num_heads sequence_length d_v")
        sequence_length = in_features.shape[-2]
        mask_tensor = torch.ones(sequence_length, sequence_length)
        mask = torch.tril(input=mask_tensor, diagonal=0).to(dtype=torch.bool)
        multihead = scaled_dot_product_attention(Q=q_transformed, K=k_transformed, V=v_transformed, mask=mask)
        collapsed_head = rearrange(multihead, "... num_heads sequence_length d_k -> ... (num_heads d_k) sequence_length")
        return einsum(self.o_proj_weight, collapsed_head, "d_out d_in, ... d_in sequence_length -> ... sequence_length d_out")
