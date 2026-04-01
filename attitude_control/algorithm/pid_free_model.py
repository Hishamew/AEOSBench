__all__ = [
    'MAX_TORQUE',
]
from copy import deepcopy
from typing import Optional

import torch
from todd.configs import PyConfig
from torch import nn

from .base import BaseModel
from .environment import Observation
from .registry import ModelRegistry
from .utils import InputNormalizer, continuous_actions_sample

DATA_DIM = 28
MAX_TORQUE = 0.2


class ExtractFeatureMixin:

    def _extract_feature(self, observation: Observation) -> torch.Tensor:
        dynamic_observation = torch.cat(
            [
                observation['reaction_wheels_speed'],  # 3
                observation['spacecraft_angular_velocity'],  # 3
                observation['spacecraft_attitude'],  # 3
                observation['battery_percentage']  # 1
            ],
            dim=-1,
        )  # 10

        static_observation = torch.cat(
            [
                observation['reaction_wheels_inertia'],  # 3
                observation['spacecraft_mass_property'],  # 4
            ],
            dim=-1,
        )  # 7

        attitude_info = observation['attitude_guide_info']
        task_time_data = observation['task_time_data']
        current_time = observation['current_time']
        extra_observation = torch.cat(
            [
                attitude_info.attitude_BR,
                attitude_info.angular_velocity_BR_B,
                attitude_info.angular_acceleration_RN_B,
                (current_time - task_time_data['start_time']).unsqueeze(-1) /
                360.,  # TODO: Cos embedding
                (current_time - task_time_data['end_time']).unsqueeze(-1) /
                360.,
            ],
            dim=-1,
        )  # 11
        return torch.cat(
            [
                dynamic_observation,
                static_observation,
                extra_observation,
            ],
            dim=-1,
        )


@ModelRegistry.register_()
class RandomSample(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, observation: Observation, *args, **kwargs) -> None:
        x = observation['battery_percentage']
        if x.dim() != 1:
            batch_size = x.size(0)
            torque = torch.randn(batch_size, 3).to(x)
        else:
            torque = torch.randn(3).to(x)

        return torch.clamp(
            torque,
            min=-MAX_TORQUE,
            max=MAX_TORQUE,
        )


class AttitudeControlMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hard_coded_max_torque: bool = True,
    ) -> None:
        super().__init__()
        self._input_projection = nn.Linear(input_dim, hidden_dim)
        self._mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self._mu_projector = nn.Linear(hidden_dim, 3)
        self._hard_coded_max_torque = hard_coded_max_torque

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        x = self._extract_features(x)
        torque = self._sample_actions(
            x,
            **kwargs,
        )

        if return_logits:
            return x, torque
        return torque

    def _extract_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self._input_projection(x)
        x = self._mlp(x)
        return x

    def _sample_actions(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        torque = self._mu_projector(x)
        if self._hard_coded_max_torque:
            torque = torch.tanh(torque) * MAX_TORQUE
        return torque


class AttitudeControlMLPStochastic(AttitudeControlMLP):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim,
            hidden_dim,
            *args,
            **kwargs,
        )
        self._log_std_projector = nn.Linear(hidden_dim, 3)

    def _sample_actions(
        self,
        x: torch.Tensor,
        deterministic: bool,
    ) -> torch.Tensor:
        torque = self._mu_projector(x)
        log_std = self._log_std_projector(x)
        if not deterministic:
            log_std = self._log_std_projector(x)
            sigma = torch.exp(torch.clamp(log_std, -20, 2))
            torque, self.epsilon = continuous_actions_sample(
                torque, sigma, return_epsilon=True)

        if self._hard_coded_max_torque:
            torque = torch.tanh(torque) * MAX_TORQUE

        return torque


@ModelRegistry.register_()
class AttitudeControlModel(BaseModel, ExtractFeatureMixin):

    def __init__(
        self,
        time_invariant: bool,
        hidden_dim: int,
        deterministic: bool = True,
        update_normalizer: bool = True,
        hard_coded_max_torque: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self._time_invariant = time_invariant
        self._hard_coded_max_torque = hard_coded_max_torque
        if time_invariant:
            input_dim = DATA_DIM - 2  # remove time info from input
        else:
            input_dim = DATA_DIM

        input_dim += 3
        self._integral = torch.zeros(3)

        if deterministic:
            self._base_model = AttitudeControlMLP(
                input_dim,
                hidden_dim,
                hard_coded_max_torque,
            )
        else:
            self._base_model = AttitudeControlMLPStochastic(
                input_dim,
                hidden_dim,
                hard_coded_max_torque,
            )
        # input_dim does not include integral dim

        self._update_normalizer = update_normalizer
        self._input_dim = input_dim
        self._input_normalizer = InputNormalizer(input_dim)
        self._runtime_normalizer = deepcopy(self._input_normalizer)

        def backward_hook(module: 'AttitudeControlModel', *args, **kwargs):
            if not module.need_update:
                return None

            module.update_normalizer()
            module.need_update = False

            return None

        self.register_full_backward_hook(backward_hook)
        self.need_update = False

    def update_normalizer(self) -> None:
        self._runtime_normalizer.load_state_dict(
            self._input_normalizer.state_dict())

    def forward(
        self,
        observation: Observation,
        deterministic: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if observation['current_time'] == 0.:
            self._integral = torch.zeros_like(
                observation['spacecraft_attitude'])
        guidance_info = observation['attitude_guide_info']
        self._integral = self._integral + guidance_info.attitude_BR.detach()

        x = self._extract_feature(observation).detach()

        # compatible with time-invariant model
        # dumping time info
        if self._time_invariant:
            x = x[..., :-2]

        x = torch.cat([x, self._integral], dim=-1)

        if self._update_normalizer:
            self._input_normalizer.update(x)
            self.need_update = True

        x = self._runtime_normalizer(x)

        return self._base_model(
            x,
            deterministic=deterministic,
        )


@ModelRegistry.register_()
class ResidualController(AttitudeControlModel):

    def __init__(
        self,
        compensation_factor: float = 0.01,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self._correction_model = deepcopy(self._base_model)
        self._compensation_factor = compensation_factor

    def forward(
        self,
        observation: Observation,
        deterministic: bool,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if observation['current_time'] == 0.:
            self._integral = torch.zeros_like(
                observation['spacecraft_attitude'])
        guidance_info = observation['attitude_guide_info']
        self._integral = self._integral + guidance_info.attitude_BR.detach()

        x = self._extract_feature(observation).detach()

        # compatible with time-invariant model
        # dumping time info
        if self._time_invariant:
            x = x[..., :-2]

        x = torch.cat([x, self._integral], dim=-1)

        if self._update_normalizer:
            self._input_normalizer.update(x)
            self.need_update = True

        x = self._runtime_normalizer(x)

        base_torque = self._base_model(
            x,
            deterministic=deterministic,
        )
        correction_torque = self._correction_model(
            x,
            deterministic=deterministic,
        )

        return base_torque + self._compensation_factor * correction_torque
