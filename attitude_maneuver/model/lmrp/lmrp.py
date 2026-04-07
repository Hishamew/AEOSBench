__all__ = [
    'LearnablePIDConfigure',
]

import torch
from torch import nn


class LearnablePIDConfigure(nn.Module):

    def __init__(
        self,
        num_sats: int,
        with_integral_limit: bool = False,
    ):
        super().__init__()
        self._with_integral_limit = with_integral_limit
        self._num_params = 4 if with_integral_limit else 3

        self._params = nn.Parameter(torch.empty(num_sats, self._num_params))

        self.register_buffer(
            '_done',
            torch.zeros(num_sats, dtype=torch.bool),
        )

        self._params.register_hook(self._frozen_params)

    @property
    def with_integral_limit(self) -> bool:
        return self._with_integral_limit

    @property
    def done(self) -> torch.Tensor:
        return self.get_buffer('_done')

    @done.setter
    def done(self, done: torch.Tensor) -> None:
        self.done.copy_(done)

    def _frozen_params(self, grad: torch.Tensor) -> torch.Tensor:
        if self.done.any():
            grad = grad.clone()
            grad[self.done] = 0.0
        return grad

    def forward(self) -> torch.Tensor:
        if self._with_integral_limit:
            return self._params.clone()

        return torch.cat(
            [
                self._params,
                torch.full_like(self._params[:, :1], 1e-3),
            ],
            dim=-1,
        )
