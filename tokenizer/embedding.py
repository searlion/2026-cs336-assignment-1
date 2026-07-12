import torch
from torch import nn
from einops import einsum


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.embedding_matrix = nn.init.trunc_normal_(self.embedding_matrix, mean=0, std=1, a=-3, b=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[x]