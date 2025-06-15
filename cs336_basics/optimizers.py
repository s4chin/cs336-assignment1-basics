from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {'lr': lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8):
        assert lr >= 0, "learning rate should be >= 0"

        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps

        defaults = {'lr': lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)

                m = self.betas[0] * m + (1 - self.betas[0]) * p.grad.data
                v = self.betas[1] * v + (1 - self.betas[1]) * ((p.grad.data) ** 2)
                lr_adjusted = lr * math.sqrt(1 - self.betas[1] ** t) / (1 - self.betas[0] ** t)

                p.data -= lr_adjusted * m / (torch.sqrt(v) + self.eps)
                p.data -= lr * self.weight_decay * p.data

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss