import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor

# Functions

def silu(x):
    return x * torch.sigmoid(x)

def softmax(x, dim):
    maxval = torch.max(x, dim=dim, keepdim=True).values
    return torch.exp(x - maxval) / torch.sum(torch.exp(x - maxval), dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"], K: Float[Tensor, " ... keys d_k"], V: Float[Tensor, " ... values d_v"], mask: Float[Tensor, " ... queries keys"] | None = None):
    d_k = torch.tensor(K.shape[-1], dtype=K.dtype, device=K.device)

    attention = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / torch.sqrt(d_k)
    if mask is not None:
        attention[~mask] = -torch.inf
    
    attention_map = softmax(attention, -1)
    return einsum(attention_map, V, "... queries keys, ... keys d_v -> ... queries d_v")

# Numerically unstable CE
def cross_entropy_bad(inputs, targets):
    sm = torch.log(softmax(inputs, dim=-1))
    sm = -sm[torch.arange(sm.shape[0]), targets]
    return torch.mean(sm)

def cross_entropy(inputs, targets):
    maxval = torch.max(inputs, dim=-1, keepdim=True).values
    log_softmax = -((inputs - maxval) - torch.log(torch.sum(torch.exp(inputs - maxval), dim=-1, keepdim=True)))
    sm = log_softmax[torch.arange(log_softmax.size(0)), targets]
    return torch.mean(sm)

# Layer classes

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weights = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(weights, 0, std, -3 * std, 3 * std)
        self.weight = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weights = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(weights, 0, 1, -3, 3)
        self.weight = nn.Parameter(weights)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sum(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt((1/self.d_model) * (rms + self.eps))

        result = einsum(x / rms, self.weights, "... d_model, d_model -> ... d_model")

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    
    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))