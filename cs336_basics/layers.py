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
        attention.masked_fill_(mask == 0, -torch.inf)
        # Below line doesn't work when mask has different n_dim than attention
        # attention[~mask] = -torch.inf
    
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
        self.weight = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sum(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt((1/self.d_model) * (rms) + self.eps)

        result = einsum(x / rms, self.weight, "... d_model, d_model -> ... d_model")

        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device: torch.device | None = None, dtype=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        position = torch.arange(0, max_seq_len, device=device, dtype=dtype)
        denom = theta ** (torch.arange(0, d_k // 2, device=device, dtype=dtype) * 2 / d_k)
        
        thetas = position.unsqueeze(1) / denom.unsqueeze(0)

        self.register_buffer("cos", torch.cos(thetas), persistent=False)
        self.register_buffer("sin", torch.sin(thetas), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        d_k = x.shape[-1]
        if token_positions is not None:
            cos = self.cos[token_positions, :]
            sin = self.sin[token_positions, :]
        else:
            cos = self.cos[:seq_len, :]
            sin = self.sin[:seq_len, :]

        x_split = rearrange(x, "... seq_len (d group) -> ... seq_len d group", group=2, d=d_k//2)
        x_even = x_split[..., 0]
        x_odd = x_split[..., 1]

        x_even_rope = cos * x_even - sin * x_odd
        x_odd_rope = sin * x_even + cos * x_odd

        x_rope = torch.stack([x_even_rope, x_odd_rope], dim=-1)
        x_rope = rearrange(x_rope, "... seq_len d group -> ... seq_len (d group)", group=2, d=d_k//2)
        return x_rope

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope: RoPE | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.device = device
        self.dtype = dtype

        self.Wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wo = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x):
        d_k = torch.tensor(self.d_model // self.num_heads)
        seq_len = x.shape[-2]

        Q = self.Wq(x)
        Q = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads, d_k=self.d_model//self.num_heads)
        K = self.Wk(x)
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads, d_k=self.d_model//self.num_heads)
        V = self.Wv(x)
        V = rearrange(V, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads, d_k=self.d_model//self.num_heads)
        
        mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device, dtype=self.dtype).to(dtype=torch.bool))

        if self.rope:
            Q = self.rope(Q)
            K = self.rope(K)

        attention_output = scaled_dot_product_attention(Q, K, V, mask)

        attention_output = rearrange(attention_output, "... h seq_len d_k -> ... seq_len (h d_k)", h=self.num_heads, d_k=self.d_model//self.num_heads)

        output = self.Wo(attention_output)
        return output

class PreNormTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.device = device
        self.dtype = dtype

        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device, dtype=dtype)
        self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype)

        self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x):
        x = x + self.attn(self.rmsnorm1(x))
        x = x + self.ff(self.rmsnorm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_heads, num_layers, d_ff, rope=None, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.rope = rope

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([PreNormTransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=rope, device=device, dtype=dtype)
                       for i in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.token_embeddings(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x