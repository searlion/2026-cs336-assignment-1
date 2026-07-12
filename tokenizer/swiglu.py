import torch
from jaxtyping import Float
from einops import einsum

class SwiGLU(torch.nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.w2_weight = torch.nn.Parameter(torch.empty(d_model, d_ff))
        self.w3_weight = torch.nn.Parameter(torch.empty(d_ff, d_model))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        silu = torch.multiply(w1_x, torch.sigmoid(w1_x))
        w3_x = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        glu = torch.multiply(silu, w3_x)
        swiglu = einsum(self.w2_weight, glu, "d_ff d_model, ... d_model -> ... d_ff")
        return swiglu


