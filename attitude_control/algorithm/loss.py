from typing import Literal

import torch
from torch import nn


def lp_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    p: float = 1.5,
    reduction: Literal['none', 'mean', 'sum'] = 'none',
):
    diff = (pred - true).abs()

    loss = diff**p

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
