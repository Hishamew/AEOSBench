__all__ = ['MLPPIDConfigure']
from copy import deepcopy

import torch
from torch import nn

from .normalize import InputNormalizer

INPUT_DIM = 7


class MLPPIDConfigure(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        with_integral_limit: bool = False,
        ki_manual_normalize: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = INPUT_DIM
        self._hidden_dim = hidden_dim
        self._output_dim = 4 if with_integral_limit else 3
        self._ki_manual_normalize = ki_manual_normalize

        self.input_projection = nn.Linear(self._input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, self._output_dim),
        )

        self._input_normalizer = InputNormalizer(self._input_dim)
        self._runtime_normalizer = deepcopy(self._input_normalizer)

        def backward_hook(
            module: 'MLPPIDConfigure',
            *args,
            **kwargs,
        ):
            if not module.need_update:
                return None

            module.update_normalizer()

            return None

        self.register_full_backward_hook(backward_hook)

        # Initialize the last layer bias to some reasonable values
        last_layer = self.mlp[-1]
        self.init_bias(
            last_layer,
            target_values=[10.0, 1e-3, 10.0, 1e-3]
            if with_integral_limit else [10.0, 1e-3, 10.0]
        )

    @property
    def need_update(self) -> bool:
        return self._runtime_normalizer.count < self._input_normalizer.count

    def update_normalizer(self) -> None:
        self._runtime_normalizer.load_state_dict(
            self._input_normalizer.state_dict()
        )

    def init_bias(self, layer: nn.Linear, target_values: list[float]) -> None:
        with torch.no_grad():
            bias_val = torch.log(torch.tensor(target_values))
            layer.bias.copy_(bias_val)

    def forward(
        self,
        sc_mass: torch.Tensor,
        sc_inertia: torch.Tensor,
        rw_inertia: torch.Tensor,
    ) -> torch.Tensor:
        feature = torch.cat(
            [
                rw_inertia,
                sc_mass.unsqueeze(-1),
                sc_inertia,
            ],
            dim=-1,
        )
        if self.training and torch.is_grad_enabled():
            self._input_normalizer.update(feature)
        feature = self._runtime_normalizer(feature)

        x = self.input_projection(feature)
        raw_gains = self.mlp(x)

        pid_gains: torch.Tensor = torch.exp(raw_gains)
        if self._output_dim == 3:
            integral_limit = torch.full_like(pid_gains[..., :1], 1e-3)
            pid_gains = torch.cat([pid_gains, integral_limit], dim=-1)

        if self._ki_manual_normalize:
            pid_gains = torch.where(
                torch.arange(self._output_dim, device=pid_gains.device) == 1,
                pid_gains * torch.tensor([1e-4], device=pid_gains.device),
                pid_gains,
            )
        return pid_gains
