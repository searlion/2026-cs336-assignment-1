import torch
from einops import einsum, rearrange

class Rope(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        positions_vector = torch.arange(0, self.max_seq_len, step=1, dtype=torch.float32)
        frequency_vector = torch.arange(0, self.d_k/2, step=1, dtype=torch.float32)
        inv_freq = self.theta **(-2 * frequency_vector / self.d_k)
        angle_table = einsum(positions_vector, inv_freq, "max_seq_len, k -> max_seq_len k")
        sin_table = torch.sin(angle_table)
        cos_table = torch.cos(angle_table)
        self.register_buffer("cos_cached", cos_table, persistent=False)
        self.register_buffer("sin_cached", sin_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        evens = x[..., 0::2]
        odds = x[..., 1::2]
        even_rotated = evens * cos - odds * sin
        odd_rotated = evens * sin + odds * cos
        interleaved = torch.stack([even_rotated, odd_rotated], dim=-1)
        return rearrange(interleaved, "... half two -> ... (half two)")
