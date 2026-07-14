import einops
import torch
from torch import Tensor
from jaxtyping import Int, Float

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    max_a = einops.reduce(inputs, pattern="batch_size vocab_size -> batch_size 1", reduction="max")
    inputs_adjusted = inputs - max_a
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_adjusted), dim=-1, keepdim=True))
    true_logit = torch.gather(input=inputs, index=targets.unsqueeze(-1), dim=-1)
    per_sample_loss = (log_sum_exp - true_logit + max_a).squeeze(-1)
    return per_sample_loss.mean()