__all__ = []
from copy import deepcopy

import torch
from torch import nn

from satsim.architecture import Timer
from satsim.attitude_control import MRPFeedback, MRPFeedbackStateDict

from .base import BaseModel
from .environment import Observation
from .pid_free_model import MAX_TORQUE
from .registry import ModelRegistry
from .utils import InputNormalizer


class BasePIDBasedModel(BaseModel):
    _controller_state_dict: MRPFeedbackStateDict

    def __init__(self) -> None:
        super().__init__()

        def pre_forward_module_reset_hook(
            module: 'BasePIDBasedModel',
            input,
        ) -> None:
            observation: Observation = input[0]
            current_time = observation['current_time']
            if current_time == 0.:
                module.reset()

        self.register_forward_pre_hook(pre_forward_module_reset_hook)

    def reset(self) -> None:
        self._controller_state_dict = MRPFeedbackStateDict(
            integral_sigma=torch.zeros(3))


class MRPFeedbackMixin:

    def _extract_mrpfeedback_input(self, observation: Observation):
        pointing_guide_output = observation['attitude_guide_info']
        wheel_speeds = observation['reaction_wheels_speed'].unsqueeze(-2)
        bs = wheel_speeds.size(0)

        diag_inertia = observation['spacecraft_mass_property'][..., 1:]
        inertia_spacecraft_point_b_in_body = torch.diag_embed(diag_inertia)

        reaction_wheels_inertia_wrt_spin = observation[
            'reaction_wheels_inertia'].unsqueeze(-2)
        reaction_wheels_spin_axis = torch.eye(3).expand(bs, 3,
                                                        3).to(wheel_speeds)

        return dict(
            sigma_BR=pointing_guide_output.attitude_BR.detach(),
            omega_BR_B=pointing_guide_output.angular_velocity_BR_B.detach(),
            omega_RN_B=pointing_guide_output.angular_velocity_RN_B.detach(),
            domega_RN_B=pointing_guide_output.angular_acceleration_RN_B.detach(
            ),
            wheel_speeds=wheel_speeds.detach(),
            inertia_spacecraft_point_b_in_body=
            inertia_spacecraft_point_b_in_body.detach(),
            reaction_wheels_inertia_wrt_spin=reaction_wheels_inertia_wrt_spin.
            detach(),
            reaction_wheels_spin_axis=reaction_wheels_spin_axis.detach(),
        )

    def _mrp_forward(
        self,
        k: torch.Tensor,
        ki: torch.Tensor,
        p: torch.Tensor,
        controller_state_dict: MRPFeedbackStateDict,
        sigma_BR: torch.Tensor,
        omega_BR_B: torch.Tensor,
        omega_RN_B: torch.Tensor,
        domega_RN_B: torch.Tensor,
        wheel_speeds: torch.Tensor,
        inertia_spacecraft_point_b_in_body: torch.Tensor,
        reaction_wheels_inertia_wrt_spin: torch.Tensor,
        reaction_wheels_spin_axis: torch.Tensor,
        integral_limit: float = 0.1,
        dt=1.,
    ) -> tuple[
            MRPFeedbackStateDict,
            torch.Tensor,
    ]:
        if k.dim() == 1:
            k = k.unsqueeze(-1)
        if ki.dim() == 1:
            ki = ki.unsqueeze(-1)
        if p.dim() == 1:
            p = p.unsqueeze(-1)

        integral_sigma = controller_state_dict['integral_sigma'].detach()

        omega_BN_B = omega_BR_B + omega_RN_B

        integral_sigma = (integral_sigma + k * dt * sigma_BR)

        clamp_mask = (torch.abs(integral_sigma) > integral_limit)

        integral_sigma = torch.clamp(
            integral_sigma,
            -integral_limit,
            integral_limit,
        )

        controller_state_dict['integral_sigma'] = integral_sigma

        attitude_error_measure = integral_sigma + torch.einsum(
            '...ij, ...j -> ...i',
            inertia_spacecraft_point_b_in_body,
            omega_BR_B,
        )

        integral_feedback_output = attitude_error_measure * ki * p  # v3_5
        attitude_control_torque = (sigma_BR * k + omega_BR_B * p +
                                   integral_feedback_output)  # Lr

        angular_momentum_BN_B = torch.einsum(
            '...ij,...j -> ...i',
            inertia_spacecraft_point_b_in_body,
            omega_BN_B,
        )
        angular_momentum = ((reaction_wheels_inertia_wrt_spin * (torch.einsum(
            '...i,...ij->...j',
            omega_BN_B,
            reaction_wheels_spin_axis,
        ).unsqueeze(-2) + wheel_speeds) * reaction_wheels_spin_axis).sum(-1) +
                            angular_momentum_BN_B)  # v3_6

        temp2 = omega_RN_B + attitude_error_measure * ki
        attitude_control_torque = attitude_control_torque + torch.cross(
            angular_momentum,
            temp2,
            dim=-1,
        )
        attitude_control_torque = attitude_control_torque + torch.einsum(
            '...ij,...j -> ...i',
            inertia_spacecraft_point_b_in_body,
            (omega_BN_B.cross(omega_RN_B, dim=-1) - domega_RN_B),
        )

        clamp_mask = attitude_control_torque.abs() > MAX_TORQUE
        max_torque = torch.where(
            attitude_control_torque > 0,
            MAX_TORQUE,
            -MAX_TORQUE,
        )
        clamped_torque = torch.where(
            clamp_mask,
            attitude_control_torque -
            (attitude_control_torque - max_torque).detach(),
            attitude_control_torque,
        )
        return controller_state_dict, -clamped_torque


@ModelRegistry.register_()
class MRPFeedbackController(BasePIDBasedModel, MRPFeedbackMixin):

    def __init__(
        self,
        dt: float,
        k: float,
        ki: float,
        p: float,
        integral_limit: float = 0.1,
    ) -> None:
        super().__init__()
        self._timer = Timer(dt)
        self.register_buffer('k', torch.tensor([k]))
        self.register_buffer('ki', torch.tensor([ki]))
        self.register_buffer('p', torch.tensor([p]))
        self._integral_limit = integral_limit
        self.reset()

    def forward(
        self,
        observation: Observation,
        *args,
        **kwargs,
    ):
        mrp_input = self._extract_mrpfeedback_input(observation)
        self._controller_state_dict, torque = self._mrp_forward(
            self.k,
            self.ki,
            self.p,
            self._controller_state_dict,
            **mrp_input,
        )
        return torque


@ModelRegistry.register_()
class MLPConfiguredPIDController(
        BasePIDBasedModel,
        MRPFeedbackMixin,
):

    def __init__(
        self,
        hidden_dim: int,
        task_invariant: bool = False,
        integral_limit: float | None = 0.1,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self._input_dim = 7 if task_invariant else 12
        self._hidden_dim = hidden_dim
        self._dt = dt
        self._integral_limit = integral_limit
        self._task_invariant = task_invariant

        self._num_params = 3 if integral_limit is not None else 4

        self.input_projection = nn.Linear(self._input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, self._num_params),
        )

        self._input_normalizer = InputNormalizer(self._input_dim)
        self._runtime_normalizer = deepcopy(self._input_normalizer)

        def backward_hook(module: 'MLPConfiguredPIDController', *args,
                          **kwargs):
            if not module.need_update:
                return None

            module.update_normalizer()
            module.need_update = False

            return None

        self.register_full_backward_hook(backward_hook)
        self.need_update = False

        # Initialize the last layer bias to some reasonable values
        last_layer = self.mlp[-1]
        self.init_bias(last_layer, target_values=[10.0, 1e-3, 10.0])

        self.reset()

    def update_normalizer(self) -> None:
        self._runtime_normalizer.load_state_dict(
            self._input_normalizer.state_dict())

    def _extract_feature(self, observation: Observation) -> torch.Tensor:

        static_observation = torch.cat(
            [
                observation['reaction_wheels_inertia'],
                observation['spacecraft_mass_property'],
            ],
            dim=-1,
        )

        if self._task_invariant:
            return static_observation

        attitude_info = observation['attitude_guide_info']
        task_time_data = observation['task_time_data']
        current_time = observation['current_time']
        task_init_observation = torch.cat(
            [
                attitude_info.attitude_BR,
                (current_time - task_time_data['start_time']).unsqueeze(-1),
                (current_time - task_time_data['end_time']).unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                static_observation,
                task_init_observation,
            ],
            dim=-1,
        )

    def init_bias(self, layer: nn.Linear, target_values: list[float]) -> None:
        with torch.no_grad():
            bias_val = torch.log(torch.tensor(target_values))
            layer.bias.copy_(bias_val)

    def forward(
        self,
        observation: Observation,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        current_time = observation['current_time']
        if current_time == 0.:
            feature = self._extract_feature(observation).detach()

            self._input_normalizer.update(feature)
            self.need_update = True
            feature = self._runtime_normalizer(feature)

            x = self.input_projection(feature)
            raw_gains = self.mlp(x)

            pid_gains: torch.Tensor = torch.exp(raw_gains)
            # if pid_gains.requires_grad:
            #     pid_gains.register_hook(
            #         lambda x: print('pid_gains', x.mean(dim=0)))
            #     raw_gains.register_hook(
            #         lambda x: torch.save(x, 'debug/raw_gains_grad.pth'))
            self._cached_pid_params = pid_gains.unbind(-1)
            # torch.save(feature, 'debug/model_input.pth')

        mrp_input = self._extract_mrpfeedback_input(observation)
        if self._integral_limit is None:
            k, ki, p, integral_limit = self._cached_pid_params
        else:
            integral_limit = self._integral_limit
            k, ki, p = self._cached_pid_params

        ki = ki * 1e-4

        self._controller_state_dict, attitude_control_torque = self._mrp_forward(
            k=k,
            ki=ki,
            p=p,
            controller_state_dict=self._controller_state_dict,
            integral_limit=0.,
            dt=self._dt,
            **mrp_input,
        )
        return attitude_control_torque


@ModelRegistry.register_('LMRP')
class LearnableMRPFeedbackController(BasePIDBasedModel, MRPFeedbackMixin):

    def __init__(
        self,
        num_sat: int,
        dt: float = 1.,
        integral_limit: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self._num_sat = num_sat
        self._dt = dt

        raw_params = torch.log(torch.tensor([2.3, 2.12e-4, 18]))
        self._raw_params = nn.Parameter(raw_params.repeat(self._num_sat, 1))
        self._integral_limit = integral_limit
        self.reset()

    def forward(
        self,
        observation: Observation,
        *args,
        env_ids: list[int] | None = None,
        **kwargs,
    ):
        raw_params = self._raw_params[env_ids] if env_ids else self._raw_params

        pid_params = torch.exp(raw_params)
        k, ki, p = pid_params.unbind(-1)

        mrp_input = self._extract_mrpfeedback_input(observation)
        self._controller_state_dict, torque = self._mrp_forward(
            k,
            ki,
            p,
            self._controller_state_dict,
            **mrp_input,
            dt=self._dt,
            integral_limit=self._integral_limit,
        )
        return torque
