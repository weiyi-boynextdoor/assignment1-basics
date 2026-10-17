import torch
from torch import Tensor
from torch.optim import Optimizer
from jaxtyping import Bool, Float, Int
from collections.abc import Callable, Iterable
from typing import Optional
import math

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(Optimizer):
    def __init__(self, params,
        lr: float | Tensor = 1e-3,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        eps: float = 1e-8
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults=defaults)

    def step(self, closure: Optional[Callable] = None):
        pass


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn(10, 10))
    opt = SGD([weights], lr=1)
    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
