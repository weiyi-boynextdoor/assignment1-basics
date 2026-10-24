import torch
from torch import nn
import numpy.typing as npt
import numpy as np
import math

def get_batch(
    dataset:npt.NDArray,
    batch_size:int,
    context_length:int,
    device:str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = np.stack([dataset[s:s + context_length] for s in starts])
    y = np.stack([dataset[s + 1:s + context_length + 1] for s in starts])

    x = torch.as_tensor(x, dtype=torch.long, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)

    return x, y
