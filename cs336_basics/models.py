import torch
from torch import nn
import math

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
        sigma = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x:torch.Tensor):
        return x @ self.W.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids:torch.Tensor):
        return self.W[token_ids]

if __name__ == "__main__":
    IN_FEATURES = 10
    OUT_FEATURES = 5
    x = torch.arange(IN_FEATURES, dtype=torch.float)
    linear = Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.float)
    for param in linear.named_parameters():
        print(param)
    y = linear(x)
    print(y)