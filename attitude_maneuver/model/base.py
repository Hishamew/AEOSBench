from abc import ABC, abstractmethod
from typing import Any, Mapping, TypeVar

import torch
from satsim.architecture import Module
from torch import nn

T = TypeVar('T', bound=Mapping[str, Any])


class BaseAttitudeController(Module[T]):

    @abstractmethod
    def forward(
        self,
        state_dict: T,
        *args,
        sigma_BR: torch.Tensor,
        omega_BR_B: torch.Tensor,
        omega_RN_B: torch.Tensor,
        domega_RN_B: torch.Tensor,
        wheel_speeds: torch.Tensor,
        inertia_spacecraft_point_b_in_body: torch.Tensor,
        reaction_wheels_inertia_wrt_spin: torch.Tensor,
        reaction_wheels_spin_axis: torch.Tensor,
        **kwargs,
    ) -> tuple[
        T,
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        pass
