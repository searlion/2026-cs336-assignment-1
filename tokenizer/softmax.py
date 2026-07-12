import torch
from jaxtyping import Float
from einops import reduce

def softmax(in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    max_v = torch.max(in_features, dim=dim, keepdim=True)
    print(max_v)
    denominator = torch.sum(torch.exp(in_features - max_v.values), dim=dim, keepdim=True)
    return torch.exp(in_features - max_v.values)/denominator
