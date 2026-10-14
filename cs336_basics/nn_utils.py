import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    inputs_max = torch.max(inputs, dim=-1, keepdim=True).values
    inputs_exp_sum = torch.sum(torch.exp(inputs - inputs_max), dim=-1)
    picked = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    logp = picked - inputs_max.squeeze(-1) - inputs_exp_sum.log()
    return -logp.mean()


if __name__ == "__main__":
    inputs = Tensor([[1.0, 2.0, 7.0], [3.0, 4.0, 8.0]])
    targets = Tensor([0, 1]).long()
    xxx = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    result = cross_entropy(inputs, targets)
    pass
