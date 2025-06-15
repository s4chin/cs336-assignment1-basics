import torch
from collections.abc import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    for p in parameters:
        if p.grad is None:
            continue
        in_dtype = p.grad.data.dtype
        norm = torch.norm(p.grad.data, 2)
        if norm < max_l2_norm:
            continue

        p.grad.data = p.grad.data * (max_l2_norm / (norm + eps))
        p.grad.data = p.grad.data.to(in_dtype)