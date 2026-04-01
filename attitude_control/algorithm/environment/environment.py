__all__ = [
    'AttitudeMaintainEnvironment',
    'LocationPointingEnvironment',
    'SpacecraftPointingEnvironment',
]
import torch

from satsim.architecture import constants
from satsim.data.orbits import elem2rv
from satsim.simulation.gravity.spice_interface import Ephemeris
from satsim.utils import sub_mrp

from ..constellation import (AttitudeMaintainConstellation,
                             LocationPointingConstellation,
                             SpacecraftPointingConstellation)
from ..constellation.constellation import (
    AttitudeMaintainConstellationStateDict,
    LocationPointingConstellationStateDict,
    SpacecraftPointingConstellationStateDict, SpacecraftPointingStateDict)
from ..data import (AttitudeMaintainTaskConfig, AttitudeMaintainTaskset,
                    LocationPointingTaskConfig, LocationPointingTaskset,
                    SpacecraftPointingTaskConfig, SpacecraftPointingTaskset)
from ..registry import EnvironmentRegistry
from .base_environment import AttitudeControlEnvironment, TaskTimeData


@EnvironmentRegistry.register_()
class AttitudeMaintainEnvironment(AttitudeControlEnvironment):
    _task_data: AttitudeMaintainTaskset
    _simulator: AttitudeMaintainConstellation

    def __init__(
        self,
        *args,
        with_window: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self._with_window = with_window

    @property
    def task_time_data(self) -> TaskTimeData:
        if self._with_window:
            start_time, end_time = self._task_data.get_time_windows().unbind(
                -1)
        else:
            start_time = torch.zeros(self._num_envs)
            end_time = torch.full((self._num_envs, ), self._episode_length)

        return TaskTimeData(
            start_time=start_time,
            end_time=end_time,
        )

    def _sample_task(self):
        self._task_data = AttitudeMaintainTaskset.sample(self.num_envs)

    def _build_simulator(self):
        position_BP_N_init, velocity_BP_N_init = elem2rv(
            constants.MU_EARTH * 1e9,
            self._orbits_data,
        )
        self._simulator = AttitudeMaintainConstellation(
            self._task_data,
            timer=self._timer,
            constellation=self._constellations_data,
            position_BP_N_init=position_BP_N_init,
            velocity_BP_N_init=velocity_BP_N_init,
            angular_velocity_BN_B_init=torch.zeros_like(position_BP_N_init),
            integrate_method='RK',
            **self._additional_build_kwargs,
        )

    def _load_taskset_config(
        self,
        taskset_config: list[AttitudeMaintainTaskConfig] | None,
    ):
        if taskset_config is None:
            self._sample_task()
            return

        self._task_data = AttitudeMaintainTaskset.from_dicts(taskset_config)

    def calculate_real_attitude_error(self) -> torch.Tensor:
        attitude_RN = self._simulator.attitude_RN
        real_attitude_BN = self._simulator_state_dict['_spacecraft']['_hub'][
            'dynamic_params']['attitude_BN']
        attitude_BR = sub_mrp(real_attitude_BN, attitude_RN)

        return attitude_BR


@EnvironmentRegistry.register_()
class LocationPointingEnvironment(AttitudeControlEnvironment):
    _task_data: LocationPointingTaskset
    _simulator: LocationPointingConstellation

    def __init__(
        self,
        initial_ephemeris_path: str,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self._initial_ephemeris: Ephemeris = torch.load(initial_ephemeris_path)

    def _sample_task(self):
        self._task_data = LocationPointingTaskset.sample(
            self._orbits_data,
            self._initial_ephemeris,
        )

    def _build_simulator(self):
        position_BP_N_init, velocity_BP_N_init = elem2rv(
            constants.MU_EARTH * 1e9,
            self._orbits_data,
        )
        self._simulator = LocationPointingConstellation(
            self._task_data,
            timer=self._timer,
            constellation=self._constellations_data,
            position_BP_N_init=position_BP_N_init,
            velocity_BP_N_init=velocity_BP_N_init,
            angular_velocity_BN_B_init=torch.zeros_like(position_BP_N_init),
            integrate_method='RK',
            **self._additional_build_kwargs,
        )

    def _load_taskset_config(
        self,
        taskset_config: list[LocationPointingTaskConfig] | None,
    ):
        if taskset_config is None:
            self._sample_task()
            return

        self._task_data = LocationPointingTaskset.from_dicts(taskset_config)

    def calculate_real_attitude_error(self):
        reference_attitude = self._simulator.reference_attitude
        attitude_RN = reference_attitude.attitude_RN
        real_attitude_BN = self._simulator_state_dict['_spacecraft']['_hub'][
            'dynamic_params']['attitude_BN']

        return sub_mrp(
            real_attitude_BN,
            attitude_RN,
        )


@EnvironmentRegistry.register_()
class UnderactuatedLocationPointingEnvironment(LocationPointingEnvironment):

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
            use_wheels_failure=True,
        )


@EnvironmentRegistry.register_()
class SpacecraftPointingEnvironment(AttitudeControlEnvironment):
    _task_data: SpacecraftPointingTaskset
    _simulator_state_dict: SpacecraftPointingConstellationStateDict

    def _sample_task(self):
        self._task_data = SpacecraftPointingTaskset.sample(self._orbits_data)

    def _build_simulator(self):
        position_BP_N_init, velocity_BP_N_init = elem2rv(
            constants.MU_EARTH * 1e9,
            self._orbits_data,
        )
        self._simulator = SpacecraftPointingConstellation(
            self._task_data,
            timer=self._timer,
            constellation=self._constellations_data,
            position_BP_N_init=position_BP_N_init,
            velocity_BP_N_init=velocity_BP_N_init,
            angular_velocity_BN_B_init=torch.zeros_like(position_BP_N_init),
            integrate_method='RK',
            **self._additional_build_kwargs,
        )

    def _load_taskset_config(
        self,
        taskset_config: list[SpacecraftPointingTaskConfig] | None,
    ):
        if taskset_config is None:
            self._sample_task()
            return

        self._task_data = SpacecraftPointingTaskset.from_dicts(taskset_config)

    def calculate_real_attitude_error(self):
        attitude_RN = self._simulator_state_dict['_spacecraft_pointing'][
            'old_attitude_RN']
        real_attitude_BN = self._simulator_state_dict['_spacecraft']['_hub'][
            'dynamic_params']['attitude_BN']

        return sub_mrp(
            real_attitude_BN,
            attitude_RN,
        )
