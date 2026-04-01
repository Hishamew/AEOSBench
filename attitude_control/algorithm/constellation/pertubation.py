__all__ = [
    'SensorErrorModel',
    'SensorErrorModelStateDict',
    'PerturbationForceTorqueDynamicParams',
    'PerturbationForceTorqueStateDict',
    'PerturbationForceTorque',
    'GravityGradientDynamicParams',
    'GravityGradientEffectorStateDict',
    'GravityGradientEffector',
    'NoisePool',
]

from typing import Callable, Never, TypedDict

import torch

from satsim.architecture import Module, constants
from satsim.simulation.base import (BackSubMatrices, BaseStateEffector,
                                    StateEffectorStateDict)
from satsim.simulation.gravity import Ephemeris
from satsim.simulation.spacecraft import SpacecraftStateDict
from satsim.utils import mrp_to_rotation_matrix


class NoisePool(TypedDict):
    transition_error: torch.Tensor
    attitude_error: torch.Tensor
    perturbation_torque: torch.Tensor
    perturbation_force: torch.Tensor


class SensorErrorModelStateDict(TypedDict):
    error_state: torch.Tensor


class SensorErrorModel(Module[SensorErrorModelStateDict]):
    _num_states: int = 6

    def __init__(
        self,
        cross_propagate: torch.Tensor,
        process_noise_matrix: torch.Tensor,
        walk_bound: torch.Tensor,
        seed: int = 0x1badcad1,
        recorded_noise: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        assert process_noise_matrix.size(-1) == process_noise_matrix.size(
            -2) == walk_bound.size(-1) == 6
        assert (walk_bound > 0.).all()

        if recorded_noise is not None:
            self.register_buffer(
                '_recorded_noise',
                recorded_noise,
                persistent=False,
            )

        num_env = cross_propagate.size(0)

        propagate_matrix = torch.eye(self._num_states).tile(num_env, 1, 1)

        cross_propagate_indices = [(0, 3), (1, 4), (2, 5)]

        att_val = torch.where(
            cross_propagate,
            self._timer.dt,
            0.,
        )
        for i, j in cross_propagate_indices:
            propagate_matrix[:, i, j] = att_val

        self.register_buffer(
            '_propagate_matrix',
            propagate_matrix,
        )
        self.register_buffer(
            '_process_noise_matrix',
            process_noise_matrix,
        )
        self.register_buffer(
            '_walk_bound',
            walk_bound,
        )

        self._gen = torch.Generator(device=torch.get_default_device())
        self._gen.manual_seed(seed)

    @property
    def propagate_matrix(self) -> torch.Tensor:
        return self.get_buffer('_propagate_matrix')

    @property
    def process_noise_matrix(self) -> torch.Tensor:
        return self.get_buffer('_process_noise_matrix')

    @property
    def walk_bound(self) -> torch.Tensor:
        return self.get_buffer('_walk_bound')

    @property
    def recorded_noise(self) -> torch.Tensor | None:
        try:
            return self.get_buffer('_recorded_noise')
        except AttributeError:
            return None

    def reset(self) -> SensorErrorModelStateDict:
        return SensorErrorModelStateDict(
            error_state=torch.zeros(self._num_states))

    def forward(
        self,
        state_dict: SensorErrorModelStateDict,
        original_state: torch.Tensor,
        state_dot: torch.Tensor,
    ) -> tuple[SensorErrorModelStateDict, tuple[torch.Tensor, torch.Tensor]]:
        num_envs = self.propagate_matrix.size(0)

        if self.recorded_noise is not None:
            rand_normal = self.recorded_noise[self._timer.step_count]
            assert rand_normal.size(0) == num_envs and rand_normal.size(
                1) == self._num_states
        else:
            rand_normal = torch.randn(
                num_envs,
                self._num_states,
                generator=self._gen,
            )

        error = torch.einsum(
            '...ij, ...j -> ...i',
            self.process_noise_matrix,
            rand_normal,
        )

        current_error_state = state_dict['error_state']

        current_error_state = torch.einsum(
            '...ij, ...j -> ...i',
            self.propagate_matrix,
            current_error_state,
        )
        current_error_state = current_error_state + error

        new_error_state = torch.clamp(
            current_error_state,
            -self.walk_bound,
            self.walk_bound,
        )
        state_dict['error_state'] = new_error_state

        return state_dict, (
            original_state + new_error_state[..., :3],
            state_dot + new_error_state[..., 3:],
        )

    @classmethod
    def sample_sensor_error(
        cls,
        num_envs: int,
        length: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        rand_normal = torch.randn(
            (length, num_envs, cls._num_states),
            generator=generator,
        )
        return rand_normal


class PerturbationForceTorqueDynamicParams(TypedDict):
    pass


class PerturbationForceTorqueStateDict(
        StateEffectorStateDict[PerturbationForceTorqueDynamicParams]):
    sampled_noise_torque: torch.Tensor
    sampled_noise_force: torch.Tensor


class PerturbationForceTorque(
        BaseStateEffector[PerturbationForceTorqueStateDict]):

    def __init__(
        self,
        num_envs: int,
        seed: int = 42,
        process_noise_factor: float = 1e-4,
        recorded_noise_force: torch.Tensor | None = None,
        recorded_noise_torque: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._gen = torch.Generator(device=torch.get_default_device())
        self._gen.manual_seed(seed)
        self._process_noise_factor = process_noise_factor
        self._num_envs = num_envs

        if recorded_noise_force is not None:
            self.register_buffer(
                '_recorded_noise_force',
                recorded_noise_force,
                persistent=False,
            )

        if recorded_noise_torque is not None:
            self.register_buffer(
                '_recorded_noise_torque',
                recorded_noise_torque,
                persistent=False,
            )

    @property
    def recorded_noise_force(self) -> torch.Tensor | None:
        try:
            return self.get_buffer('_recorded_noise_force')
        except AttributeError:
            return None

    @property
    def recorded_noise_torque(self) -> torch.Tensor | None:
        try:
            return self.get_buffer('_recorded_noise_torque')
        except AttributeError:
            return None

    def reset(self) -> PerturbationForceTorqueStateDict:
        state_dict = super().reset()
        state_dict.update(dynamic_params=dict())

        return state_dict

    def forward(
        self,
        state_dict: PerturbationForceTorqueStateDict,
    ) -> tuple[PerturbationForceTorqueStateDict, tuple]:
        if self.recorded_noise_force is not None:
            noise_force = self.recorded_noise_force[self._timer.step_count]
            assert noise_force.size(0) == self._num_envs and noise_force.size(
                1) == 3
        else:
            noise_force = torch.randn((self._num_envs, 3), generator=self._gen)

        if self.recorded_noise_torque is not None:
            noise_torque = self.recorded_noise_torque[self._timer.step_count]
            assert noise_torque.size(
                0) == self._num_envs and noise_torque.size(1) == 3
        else:
            noise_torque = torch.randn((self._num_envs, 3),
                                       generator=self._gen)

        state_dict[
            'sampled_noise_force'] = noise_force * self._process_noise_factor
        state_dict[
            'sampled_noise_torque'] = noise_torque * self._process_noise_factor
        return state_dict, tuple()

    def update_back_substitution_contribution(
        self,
        state_dict: PerturbationForceTorqueStateDict,
        integrate_time_step: float,
        back_substitution_contribution: BackSubMatrices,
        *args,
        **kwargs,
    ) -> BackSubMatrices:
        ext_torque = back_substitution_contribution['ext_torque_B_B']
        pertubation_torque = state_dict['sampled_noise_torque']
        ext_torque = ext_torque + pertubation_torque
        back_substitution_contribution['ext_torque_B_B'] = ext_torque

        ext_force = back_substitution_contribution['ext_force_B_B']
        pertubation_force = state_dict['sampled_noise_force']
        ext_force = ext_force + pertubation_force
        back_substitution_contribution['ext_force_B_B'] = ext_force
        return back_substitution_contribution

    def compute_derivatives(
        self,
        state_dict,
        integrate_time_step,
        *args,
        **kwargs,
    ):
        return PerturbationForceTorqueDynamicParams()

    @staticmethod
    def sample_perturbation(
        num_envs: int,
        length: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise_force = torch.randn((length, num_envs, 3), generator=generator)
        noise_torque = torch.randn((length, num_envs, 3), generator=generator)
        return noise_force, noise_torque


class GravityGradientDynamicParams(TypedDict):
    pass


class GravityGradientEffectorStateDict(
        StateEffectorStateDict[GravityGradientDynamicParams]):
    pass


class GravityGradientEffector(
        BaseStateEffector[GravityGradientEffectorStateDict]):

    def reset(self) -> GravityGradientEffectorStateDict:
        state_dict = super().reset()
        state_dict.update(dynamic_params=dict())
        return state_dict

    def forward(
        self,
        state_dict: GravityGradientEffectorStateDict,
        earth_ephemeris: Ephemeris,
    ) -> GravityGradientEffectorStateDict:
        raise NotImplementedError

    def update_back_substitution_contribution(
        self,
        state_dict: GravityGradientEffectorStateDict,
        back_substitution_contribution: BackSubMatrices,
        integrate_time_step: float,
        spacecraft_state_dict: SpacecraftStateDict,
    ) -> BackSubMatrices:

        spacecraft_inertia_matrix = spacecraft_state_dict['mass_props'][
            'moment_of_inertia_matrix_wrt_body_point']
        attitude_BN = spacecraft_state_dict['_hub']['dynamic_params'][
            'attitude_BN']
        position_BP_N = spacecraft_state_dict['_hub']['dynamic_params'][
            'position_BP_N']

        dcm_BN = mrp_to_rotation_matrix(attitude_BN)
        distance_BP = position_BP_N.norm(dim=-1, keepdim=True)
        direction_BP_N = torch.nn.functional.normalize(
            position_BP_N,
            dim=-1,
        )
        direction_BP_B = torch.einsum(
            '...ij, ...j -> ...i',
            dcm_BN,
            direction_BP_N,
        )
        gradient_torque_B_B = 3.0 * constants.MU_EARTH * 1e9 / (
            distance_BP**3) * torch.einsum(
                '...ij, ...j -> ...i',
                spacecraft_inertia_matrix,
                direction_BP_B,
            )
        gradient_torque_B_B = direction_BP_B.cross(
            gradient_torque_B_B,
            dim=-1,
        )

        back_substitution_contribution[
            'ext_torque_B_B'] = back_substitution_contribution[
                'ext_torque_B_B'] + gradient_torque_B_B.detach()

        return back_substitution_contribution

    def compute_derivatives(
        self,
        state_dict: GravityGradientEffectorStateDict,
        integrate_time_step: float,
        hub_state_dot: dict[str, torch.Tensor],
        spacecraft_state_dict: SpacecraftStateDict,
    ) -> GravityGradientDynamicParams:
        return GravityGradientDynamicParams()
