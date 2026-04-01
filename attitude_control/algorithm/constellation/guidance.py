__all__ = [
    "GuidanceOutput",
    "SimpleInertial3DGuideStateDict",
    "SimpleInertial3DGuide",
    "SpacecraftPointingStateDict",
    "SpacecraftPointing",
    "att_tracking_error",
]

from typing import TypedDict

import torch
import torch.nn.functional as F

from satsim.architecture import Module
from satsim.attitude_guidance import GuidanceOutput
from satsim.utils import Bmat, dcm_to_mrp, mrp_to_rotation_matrix, sub_mrp


def att_tracking_error(
    attitude_RN: torch.Tensor,
    angular_velocity_RN_N: torch.Tensor,
    angular_acceleration_RN_N: torch.Tensor,
    attitude_BN: torch.Tensor,
    angular_velocity_BN_B: torch.Tensor,
) -> GuidanceOutput:
    attitude_BR = sub_mrp(attitude_BN, attitude_RN)

    direction_cosine_matrix_BN = mrp_to_rotation_matrix(attitude_BN)

    angular_velocity_RN_B = torch.einsum(
        '...ij,...j->...i',
        direction_cosine_matrix_BN,
        angular_velocity_RN_N,
    )
    angular_acceleration_RN_B = torch.einsum(
        '...ij,...j->...i',
        direction_cosine_matrix_BN,
        angular_acceleration_RN_N,
    )

    angular_velocity_BR_B = angular_velocity_BN_B - angular_velocity_RN_B
    return GuidanceOutput(
        attitude_BR,
        angular_velocity_BR_B,
        angular_velocity_RN_B,
        angular_acceleration_RN_B,
    )


class SimpleInertial3DGuideStateDict(TypedDict):
    pass


class SimpleInertial3DGuide(Module[SimpleInertial3DGuideStateDict]):

    def __init__(self, attitude_RN: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        to_shadow = attitude_RN.norm(dim=-1, keepdim=True) > 1
        attitude_RN = torch.where(
            to_shadow,
            -attitude_RN / (attitude_RN**2).sum(dim=-1, keepdim=True),
            attitude_RN,
        )
        self.register_buffer(
            '_attitude_RN',
            attitude_RN,
        )

    @property
    def attitude_RN(self) -> torch.Tensor:
        return self.get_buffer('_attitude_RN')

    def forward(
        self,
        state_dict: SimpleInertial3DGuideStateDict,
        attitude_BN: torch.Tensor,
        angular_velocity_BN_B: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[SimpleInertial3DGuideStateDict, GuidanceOutput]:
        attitude_BR = sub_mrp(attitude_BN, self.attitude_RN)
        angular_velocity_BR_B = angular_velocity_BN_B

        return state_dict, GuidanceOutput(
            attitude_BR,
            angular_velocity_BR_B,
            torch.zeros_like(angular_velocity_BN_B),
            torch.zeros_like(angular_velocity_BN_B),
        )


class SpacecraftPointingStateDict(TypedDict):
    old_attitude_RN: torch.Tensor
    old_angular_velocity_RN_N: torch.Tensor


class SpacecraftPointing(Module[SpacecraftPointingStateDict]):

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        # To simplify the case, we assume pointing_direction_B_B
        # is [1,0,0], which leaves nothing to do here
        super().__init__(*args, **kwargs)

    def reset(self) -> SpacecraftPointingStateDict:
        return SpacecraftPointingStateDict(
            old_angular_velocity_RN_N=torch.zeros(
                1,
                3,
            ),
            old_attitude_RN=torch.zeros(1, 3),
        )

    def forward(
        self,
        state_dict: SpacecraftPointingStateDict,
        position_BN_N: torch.Tensor,
        position_CN_N: torch.Tensor,
        attitude_BN: torch.Tensor,
        angular_velocity_BN_B: torch.Tensor,
    ):
        # NOTE: In this module, chief spacecraft is marked as C, deputy as D

        position_DN_N = position_BN_N
        position_CD_N = position_CN_N - position_DN_N

        direction_CD_N = F.normalize(position_CD_N, dim=-1)
        attitude_RN_x = direction_CD_N

        temp_z = torch.tensor([[0., 0., 1.]]).to(attitude_RN_x)
        attitude_RN_y = torch.cross(temp_z, attitude_RN_x, dim=-1)

        redirect_mask = torch.norm(attitude_RN_y, dim=-1, keepdim=True) < 1e-6
        attitude_RN_y = torch.where(
            redirect_mask,
            torch.cross(
                attitude_RN_x,
                torch.tensor([[0., 1., 0.]]).to(attitude_RN_x),
                dim=-1,
            ),
            attitude_RN_y,
        )

        attitude_RN_y = F.normalize(attitude_RN_y, dim=-1)
        attitude_RN_z = torch.cross(attitude_RN_x, attitude_RN_y, dim=-1)
        attitude_RN_z = F.normalize(attitude_RN_z, dim=-1)

        dcm_RN = torch.stack(
            [attitude_RN_x, attitude_RN_y, attitude_RN_z],
            dim=-2,
        )
        attitude_RN = dcm_to_mrp(dcm_RN)

        old_attitude_RN = state_dict['old_attitude_RN']
        delta_attitude_RN = attitude_RN - old_attitude_RN

        old_attitude_RN_shadow = -old_attitude_RN / (
            old_attitude_RN * old_attitude_RN).sum(-1, keepdim=True)
        delta_attitude_RN_shadow = attitude_RN - old_attitude_RN_shadow

        use_shadow_mask = (torch.norm(delta_attitude_RN_shadow, dim=-1)
                           < torch.norm(delta_attitude_RN, dim=-1))
        delta_attitude_RN = torch.where(
            use_shadow_mask.unsqueeze(-1),
            delta_attitude_RN_shadow,
            delta_attitude_RN,
        )
        old_attitude_RN = torch.where(
            use_shadow_mask.unsqueeze(-1),
            old_attitude_RN_shadow,
            old_attitude_RN,
        )

        attitude_RN_dot = (1 / self._timer.dt) * delta_attitude_RN

        attitude_RN_squared = torch.sum(attitude_RN**2, dim=-1)
        old_attitude_RN_squared = torch.sum(old_attitude_RN**2, dim=-1)

        old_attitude_RN_Bmat = Bmat(old_attitude_RN)
        attitude_RN_Bmat = Bmat(attitude_RN)

        attitude_RN_Bmat = (old_attitude_RN_Bmat + attitude_RN_Bmat) / 2
        attitude_RN_Bmat_transpose = torch.transpose(attitude_RN_Bmat, -1, -2)

        old_Binv_scale = (1.0 / (
            (1 + old_attitude_RN_squared)**2)).unsqueeze(-1).unsqueeze(-1)
        new_Binv_scale = (
            1.0 / ((1 + attitude_RN_squared)**2)).unsqueeze(-1).unsqueeze(-1)
        average_scale = (old_Binv_scale + new_Binv_scale) / 2

        attitude_RN_Binv = attitude_RN_Bmat_transpose * average_scale
        angular_velocity_RN_R = 4 * torch.einsum(
            '...ij,...j->...i',
            attitude_RN_Binv,
            attitude_RN_dot,
        )

        dcm_NR = torch.transpose(dcm_RN, -1, -2)
        angular_velocity_RN_N = torch.einsum(
            '...ij,...j->...i',
            dcm_NR,
            angular_velocity_RN_R,
        )

        old_angular_velocity_RN_N = state_dict['old_angular_velocity_RN_N']
        angular_velocity_RN_N_dot = (
            angular_velocity_RN_N - old_angular_velocity_RN_N) / self._timer.dt
        state_dict['old_attitude_RN'] = attitude_RN
        state_dict['old_angular_velocity_RN_N'] = angular_velocity_RN_N

        if self._timer.step_count < 2:
            angular_velocity_RN_N = torch.zeros_like(angular_velocity_RN_N)
        if self._timer.step_count < 3:
            angular_velocity_RN_N_dot = torch.zeros_like(
                angular_velocity_RN_N_dot)

        # As We assume the antenna points to axis x in body frame,
        # attitude_RN is attitude_R1N

        return state_dict, att_tracking_error(
            attitude_RN,
            angular_velocity_RN_N,
            angular_velocity_RN_N_dot,
            attitude_BN,
            angular_velocity_BN_B,
        )
