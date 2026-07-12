import torch
from einops import reduce

class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps 
        self.g = torch.nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
         in_dtype = x.dtype
         x = x.to(torch.float32)
         x_sum_squared = reduce(x**2, "... d_model-> ... 1", "sum")
         rms_a = torch.sqrt(1/self.d_model * x_sum_squared + self.eps)
         return (x/rms_a * self.g).to(in_dtype)
