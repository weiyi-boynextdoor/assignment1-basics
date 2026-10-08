import torch
from torch import nn
from einops import einsum, rearrange
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(out_features, in_features, device=device, dtype=dtype))
        sigma = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x:Tensor):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids:Tensor):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x:Tensor):
        # upcast input to torch.float32
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((einsum(x, x, "... x, ... x -> ...")) / x.shape[-1] + self.eps)
        result = x / rms.unsqueeze(-1) * self.weight
        return result.to(in_type)


class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff <= 0:
            d_ff = d_model * 8 / 3
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x:Tensor):
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        w3_x = self.w3(x)
        return self.w2(silu * w3_x)


class RoPE(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        super().__init__()
        half_d_k = d_k // 2
        exponent = torch.arange(half_d_k, dtype=torch.float)
        exponent /= half_d_k
        inv_freq = torch.pow(1.0 / theta, exponent)
        thetas = torch.outer(torch.arange(max_seq_len, dtype=torch.float), inv_freq) # shape: max_seq_len, d_k / 2
        self.register_buffer("cos_thetas", thetas.cos(), persistent=False)
        self.register_buffer("sin_thetas", thetas.sin(), persistent=False)

    def forward(self, x:Tensor, token_positions:Tensor):
        cos_thetas = self.cos_thetas[token_positions]
        sin_thetas = self.sin_thetas[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        result = torch.empty_like(x)
        result[..., 0::2] = x_even * cos_thetas - x_odd * sin_thetas
        result[..., 1::2] = x_even * sin_thetas + x_odd * cos_thetas
        return result


def softmax(x:Tensor, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None
) -> Float[Tensor, " ... queries d_v"]:
    qkt = einsum(Q, K, "... n k, ... m k -> ... n m")
    qkt /= math.sqrt(float(Q.shape[-1]))
    if mask is not None:
        qkt = qkt.masked_fill(~mask, float("-inf"))
    scores = softmax(qkt) @ V
    return scores


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, rope_theta=0, max_seq_len=0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads # let dk = dv = d_model / num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        # rope
        if rope_theta > 0 and max_seq_len > 0:
            self.rope = RoPE(rope_theta, self.d_head, max_seq_len)
        else:
            self.rope = None

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor=None):
        sequence_length = x.shape[-2]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "... seq (num_heads d_head) -> ... num_heads seq d_head", num_heads=self.num_heads)
        K = rearrange(K, "... seq (num_heads d_head) -> ... num_heads seq d_head", num_heads=self.num_heads)
        V = rearrange(V, "... seq (num_heads d_head) -> ... num_heads seq d_head", num_heads=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            Q = self.rope(Q, token_positions.unsqueeze(-2))
            K = self.rope(K, token_positions.unsqueeze(-2))

        mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device))
        attention = scaled_dot_product_attention(Q, K, V, mask)
        attention = rearrange(attention, "... num_heads seq d_head -> ... seq (num_heads d_head)", num_heads=self.num_heads)
        return self.output_proj(attention)


class PreNormTransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, theta:int, max_seq_len:int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x:torch.Tensor):
        layer1 = self.attn(self.ln1(x)) + x
        layer2 = self.ffn(self.ln2(layer1)) + layer1
        return layer2


class TransformerLM(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, d_model:int, num_layers:int, num_heads:int, d_ff:int, rope_theta:float):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        layers = []
        for i in range(num_layers):
            layers.append(PreNormTransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length))
        self.layers = nn.ModuleList(layers)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(vocab_size, d_model)

    def forward(self, in_indices:torch.Tensor):
        result = self.token_embeddings(in_indices)
        for layer in self.layers:
            result = layer(result)
        result = self.ln_final(result)
        return self.lm_head(result)


if __name__ == "__main__":
    rope = RoPE(10000.0, 100, 200)