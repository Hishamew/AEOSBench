__all__ = [
    'LearnableMRPControl',
    'LearnableMRPControlStateDict',
]
from typing import TypedDict

import einops
import torch
from satsim.architecture import Module


class LearnableMRPControlStateDict(TypedDict):
    integral_sigma: torch.Tensor
    k: torch.Tensor
    ki: torch.Tensor
    p: torch.Tensor
    integral_limit: torch.Tensor


class LearnableMRPControl(Module[LearnableMRPControlStateDict]):

    def reset(self) -> LearnableMRPControlStateDict:
        return LearnableMRPControlStateDict(integral_sigma=torch.zeros(3))

    def forward(
        self,
        state_dict: LearnableMRPControlStateDict,
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
        LearnableMRPControlStateDict,
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        integral_sigma = state_dict['integral_sigma']

        k = einops.rearrange(state_dict['k'], 'i -> i 1')
        ki = einops.rearrange(state_dict['ki'], 'i -> i 1')
        p = einops.rearrange(state_dict['p'], 'i -> i 1')
        integral_limit = einops.rearrange(
            state_dict['integral_limit'], 'i -> i 1'
        )

        omega_BN_B = omega_BR_B + omega_RN_B

        dt = 0.0 if self._timer.step_count == 0 else self._timer.dt
        integral_sigma = (integral_sigma + k * dt * sigma_BR)

        integral_sigma = torch.clamp(
            integral_sigma,
            -integral_limit,
            integral_limit,
        )
        state_dict['integral_sigma'] = integral_sigma

        attitude_error_measure = integral_sigma + torch.einsum(
            '...ij, ...j -> ...i',
            inertia_spacecraft_point_b_in_body,
            omega_BR_B,
        )

        integral_feedback_output = attitude_error_measure * ki * p  # v3_5
        attitude_control_torque = (
            sigma_BR * k + omega_BR_B * p + integral_feedback_output
        )  # Lr

        angular_momentum_BN_B = torch.einsum(
            '...ij,...j -> ...i',
            inertia_spacecraft_point_b_in_body,
            omega_BN_B,
        )
        angular_momentum = ((
            reaction_wheels_inertia_wrt_spin * (
                torch.einsum(
                    '...i,...ij->...j',
                    omega_BN_B,
                    reaction_wheels_spin_axis,
                ).unsqueeze(-2) + wheel_speeds
            ) * reaction_wheels_spin_axis
        ).sum(-1) + angular_momentum_BN_B)  # v3_6

        temp2 = omega_RN_B + attitude_error_measure * ki  # v3_8
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

        return state_dict, (
            -attitude_control_torque,
            -integral_feedback_output,
        )
