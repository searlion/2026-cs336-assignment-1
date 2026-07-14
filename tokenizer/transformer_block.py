import torch
from torch import Tensor
from jaxtyping import Float
from tokenizer import multihead_self_attention
from tokenizer import rmsnorm
from tokenizer import swiglu

class Transformer_block(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, weights: dict[str, Tensor], eps: float = 1e-5):
        super().__init__()
        self.multihead_self_attention = multihead_self_attention.Multihead_self_attention_with_rope(
            d_model=d_model, 
            num_heads=num_heads, 
            q_proj_weight=weights['attn.q_proj.weight'],
            k_proj_weight=weights['attn.k_proj.weight'],
            v_proj_weight=weights['attn.v_proj.weight'],
            o_proj_weight=weights['attn.output_proj.weight'],
            theta=theta,
            max_seq_len=max_seq_len
        )
        self.rmsnorm_1 = rmsnorm.RMSNorm(d_model, eps)
        self.rmsnorm_1.load_state_dict({'g': weights['ln1.weight']})
        self.rmsnorm_2 = rmsnorm.RMSNorm(d_model, eps)
        self.rmsnorm_2.load_state_dict({'g': weights['ln2.weight']})
        self.swiglu = swiglu.SwiGLU(d_model=d_model,d_ff=d_ff)
        self.swiglu.load_state_dict({
            'w1_weight': weights['ffn.w1.weight'],
            'w2_weight': weights['ffn.w2.weight'],
            'w3_weight': weights['ffn.w3.weight']
        })
    

    def forward(self, in_features: Float[Tensor, " batch sequence_length d_model"]):
        input_features_rmsnorm = self.rmsnorm_1.forward(in_features)
        input_features_multihead_self_attention = self.multihead_self_attention.forward(input_features_rmsnorm)
        output_self_attention = in_features + input_features_multihead_self_attention
        input_features_ffn_rmsnorm = self.rmsnorm_2.forward(output_self_attention)
        input_features_ffn = self.swiglu.forward(input_features_ffn_rmsnorm)
        output = output_self_attention + input_features_ffn
        return output