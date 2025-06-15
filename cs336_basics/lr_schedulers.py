import torch
import math


def cosine_annealing_lr(t, max_lr, min_lr, tw, tc):
    if t < tw:
        return (t / tw) * max_lr
    if t >= tw and t <= tc:
        return min_lr + 0.5 * (1 + math.cos((t - tw) / (tc - tw) * math.pi)) * (max_lr - min_lr)
    if t > tc:
        return min_lr
    assert False, "Shouldnt reach here"