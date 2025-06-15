import torch
from collections.abc import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    all_norms = []
    for p in parameters:
        if p.grad is None:
            continue
        in_dtype = p.grad.data.dtype
        norm = torch.norm(p.grad.data, 2)
        all_norms.append(norm ** 2)
    
    total_norm = torch.sqrt(sum(all_norms))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data = p.grad.data * scale
            p.grad.data = p.grad.data.to(in_dtype)