import torch
from torch import nn

class Linear(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype), requires_grad=True)

    def forward(self, x:torch.Tensor):
        return x @ self.W.T

if __name__ == "__main__":
    IN_FEATURES = 10
    OUT_FEATURES = 5
    x = torch.arange(IN_FEATURES, dtype=torch.float)
    linear = Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.float)
    for param in linear.named_parameters():
        print(param)
    y = linear(x)
    print(y)