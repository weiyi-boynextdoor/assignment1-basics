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
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults=defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for i, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if not state or "m" not in state:
                    # first moment vector
                    state["m"] = torch.zeros_like(p)
                    # second moment vector
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 0

                state["t"] += 1
                t = state["t"]
                grad = p.grad.data

                adjusted_lr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr * weight_decay * p.data

                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * (grad.square())
                p.data -= adjusted_lr * state["m"] / (state["v"].sqrt() + eps)

        return loss


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate
    ret = min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    return ret


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn(10, 10))
    opt = SGD([weights], lr=1)
    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
