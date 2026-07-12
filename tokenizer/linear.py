import torch
import numpy as np
from torch.nn.init import trunc_normal_
from torch.nn import Parameter
from einops import einsum
import math

class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features))
        variance = 2 / (self.in_features + self.out_features)
        standard_deviation = math.sqrt(variance)
        self.weight = trunc_normal_(tensor=self.weight, mean=0, std = standard_deviation, a = -3 * standard_deviation, b = 3 * standard_deviation)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")