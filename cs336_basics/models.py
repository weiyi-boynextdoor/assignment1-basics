import torch
from torch import nn
from einops import einsum
import math

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(out_features, in_features, device=device, dtype=dtype))
        sigma = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x:torch.Tensor):
        return x @ self.weights.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids:torch.Tensor):
        return self.weights[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x:torch.Tensor):
        # upcast input to torch.float32
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((einsum(x, x, "... x, ... x -> ...")) / x.shape[-1] + self.eps)
        result = x / rms.unsqueeze(-1) * self.weights
        return result.to(in_type)


class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff <= 0:
            d_ff = d_model * 8 / 3
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.randn(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.randn(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.randn(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x:torch.Tensor):
        w1_x = x @ self.w1.T
        silu = w1_x * torch.sigmoid(w1_x)
        w3_x = x @ self.w3.T
        return (silu * w3_x) @ self.w2.T


if __name__ == "__main__":
    IN_FEATURES = 10
    OUT_FEATURES = 5
    x = torch.arange(IN_FEATURES, dtype=torch.float)
    linear = Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.float)
    for param in linear.named_parameters():
        print(param)
    y = linear(x)
    print(y)