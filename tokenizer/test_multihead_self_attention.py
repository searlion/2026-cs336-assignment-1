from tokenizer.multihead_self_attention import Multihead_self_attention
import torch

multihead_self_attention = Multihead_self_attention(
    d_model=8, 
    num_heads=2, 
    q_proj_weight=torch.randn(8,8), 
    k_proj_weight=torch.randn(8,8), 
    v_proj_weight=torch.randn(8,8), 
    o_proj_weight=torch.randn(8,8), 
)

multihead_self_attention.forward(
    in_features=torch.randn(1,3,8)
)