from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas: tuple[float, float] = (0.9, 0.95), weight_decay: float = 0.01, eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta_1 = group["betas"][0]
            beta_2 = group["betas"][1]
            weight_decay = group["weight_decay"]
            # lr = lr * math.sqrt(1 - beta_2) / (1 - beta_1)
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data                                              # compute gradient of the loss
                alpha_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)       # compute adjusted alpha for iteration t
                p.data = p.data - lr * weight_decay * p.data                           # apply weight decay
                m = beta_1 * m + (1 - beta_1) * grad                            # update the first moment estimate
                v = beta_2 * v + (1 - beta_2) * grad**2                         # update the second moment estaimte
                p.data -= alpha_t / torch.sqrt(v + eps) * m                     # apply the moment-adjusted weight updates
                t = t + 1
                state["t"] = t 
                state["m"] = m
                state["v"] = v
        
        return loss

weights = torch.nn.Parameter(5 * torch.randn((10,10)))
opt = AdamW([weights], lr = 1e-1)

for t in range(100):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()

