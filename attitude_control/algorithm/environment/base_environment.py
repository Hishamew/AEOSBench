__all__ = [
    'AttitudeControlEnvironment',
    'TaskTimeData',
    'StaticObservation',
    'Observation',
    'Loss',
    'TaskConfig',
    'Taskset',
    'BuildConfig',
]
import os
import random
from abc import ABC, abstractmethod
from typing import NotRequired, TypedDict, Union

import torch
from todd.loggers import master_logger
from todd.patches.py_ import json_load

from satsim.architecture import Timer
from satsim.data import OrbitalElements, OrbitDict
from satsim.simulation.gravity import Ephemeris
from satsim.utils import dict_recursive_apply, move_to
from satsim.utils.matrix_support import Binv

from ..constellation import (BaseConstellation, BaseConstellationStateDict,
                             DynamicObservation,
                             LocationPointingConstellationStateDict)
from ..data import (AttitudeMaintainTaskConfig, AttitudeMaintainTaskset,
                    Constellation, LocationPointingTaskConfig,
                    LocationPointingTaskset, SatelliteConfig,
                    SpacecraftPointingTaskConfig, SpacecraftPointingTaskset)
from ..loss import lp_loss


class TaskTimeData(TypedDict):
    start_time: torch.Tensor
    end_time: torch.Tensor


class StaticObservation(TypedDict):
    reaction_wheels_inertia: torch.Tensor
    spacecraft_mass_property: torch.Tensor
    task_time_data: TaskTimeData
    wheels_failure: torch.IntTensor


class Observation(DynamicObservation, StaticObservation):
    pass


TaskConfig = Union[
    LocationPointingTaskConfig,
    AttitudeMaintainTaskConfig,
    SpacecraftPointingTaskConfig,
]
Taskset = Union[
    LocationPointingTaskset,
    AttitudeMaintainTaskset,
    SpacecraftPointingTaskset,
]


class Loss(TypedDict):
    attitude_loss: torch.Tensor
    battery_loss: torch.Tensor
    loss_mask: torch.Tensor
    nominal_loss: torch.Tensor
    motion_loss: torch.Tensor


class BuildConfig(TypedDict):
    constellation: list[SatelliteConfig]
    orbits: NotRequired[list[OrbitDict]]
    tasks: NotRequired[list[TaskConfig]]


class AttitudeControlEnvironment(ABC):
    _task_data: Taskset
    _simulator: BaseConstellation
    _simulator_state_dict: BaseConstellationStateDict

    def __init__(
        self,
        num_envs: int,
        train_dir: str | None = None,
        build_config: BuildConfig | None = None,
        episode_length: int = 360,
        **kwargs,
    ) -> None:
        self._num_envs = num_envs
        self._timer = Timer(1.)
        self._episode_length = episode_length

        self._additional_build_kwargs = kwargs

        if isinstance(build_config, dict):
            self.load_build_config(build_config)
        elif isinstance(build_config, str):
            self.load_build_config(json_load(build_config))

        if train_dir:
            master_logger.info("Orbits will be sampled from given pool")
            self._train_orbits_pool: list[OrbitDict] = json_load(
                os.path.join(train_dir, 'orbits.json'))
        else:
            master_logger.info("Orbits will be sampled randomly")
            self._train_orbits_pool = None

    @property
    def task_time_data(self) -> TaskTimeData:
        return TaskTimeData(
            start_time=torch.zeros(self._num_envs),
            end_time=torch.full(
                (self._num_envs, ),
                self._episode_length,
            ),
        )

    @property
    def simulator_state_dict(self) -> BaseConstellationStateDict:
        return self._simulator_state_dict

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def has_initialized(self) -> bool:
        return hasattr(self, '_simulator')

    @property
    def build_config(self) -> BuildConfig:
        return BuildConfig(
            orbits=self._orbits_data.to_dicts(),
            constellation=self._constellations_data.to_dicts(),
            tasks=self._task_data.to_dicts(),
        )

    def _sample_constellation(self) -> None:
        self._constellations_data = Constellation.sample(self._num_envs)
        # rw.power is deserted and will be hard coded to zero

    def _sample_orbits(self) -> None:
        if self._train_orbits_pool:
            sampled_orbits = random.sample(
                self._train_orbits_pool,
                self._num_envs,
            )
            for orbit in sampled_orbits:
                orbit['true_anomaly'] = round(
                    random.uniform(0, 2) * torch.pi, 6)

            orbits = OrbitalElements.from_dicts(sampled_orbits)
        else:
            orbits = OrbitalElements.sample(self._num_envs)
        self._orbits_data = orbits

    @abstractmethod
    def _sample_task(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _build_simulator(self) -> None:
        raise NotImplementedError

    def _stochastic_init(
        self,
        sample_constellation: bool = True,
        sample_task: bool = True,
        **kwargs,
    ) -> None:
        if sample_constellation:
            self._sample_constellation()

        if sample_task:
            self._sample_orbits()
            self._sample_task()

        if sample_task or sample_constellation:
            self._build_simulator()

    def clear_grad(self) -> None:
        self._simulator_state_dict = dict_recursive_apply(
            self._simulator_state_dict,
            lambda x: x.clone().detach(),
        )
        self._last_attitude_BR = self._last_attitude_BR.detach()

    def reset(
        self,
        **kwargs,
    ) -> tuple[Observation, Loss]:
        self._stochastic_init(**kwargs)
        self._timer.reset()
        self._simulator_state_dict = self._simulator.reset()
        if hasattr(self, '_last_attitude_BR'):
            del self._last_attitude_BR

        return self.step(torch.zeros(self._num_envs, 3))

    def _pick_static_data(self) -> StaticObservation:
        if hasattr(self, '_static_data_buffer'):
            return getattr(self, '_static_data_buffer')

        reaction_wheels_inertia = self._simulator.reaction_wheels.moment_of_inertia_wrt_spin.squeeze(
            -2)
        if reaction_wheels_inertia.size(0) != self._num_envs:
            reaction_wheels_inertia = reaction_wheels_inertia.expand(
                self._num_envs, -1)

        spacecraft_mass_property = self._constellations_data.to_tensor()
        if spacecraft_mass_property.size(0) != self._num_envs:
            spacecraft_mass_property = spacecraft_mass_property.expand(
                self._num_envs, -1)

        self._static_data_buffer = StaticObservation(
            reaction_wheels_inertia=reaction_wheels_inertia,
            spacecraft_mass_property=spacecraft_mass_property,
            task_time_data=self.task_time_data,
        )
        if isinstance(self._task_data, LocationPointingTaskset):
            reaction_wheels_failure = torch.tensor(
                [t.broken_wheels for t in self._task_data])
            self._static_data_buffer.update(
                wheels_failure=reaction_wheels_failure)

        return self._pick_static_data()

    @abstractmethod
    def calculate_real_attitude_error(self) -> torch.Tensor:
        pass

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[
            Observation,
            Loss,
    ]:
        self._earth_ephemeris: Ephemeris
        dynamic_observation: DynamicObservation
        self._simulator_state_dict, (
            dynamic_observation,
            self._earth_ephemeris,
        ) = self._simulator(
            self._simulator_state_dict,
            torque=actions,
        )
        observation = Observation(
            **dynamic_observation,
            **self._pick_static_data(),
        )

        battery_percentage = observation['battery_percentage']
        battery_loss = (0.8 - battery_percentage).sum() / self._num_envs

        loss_mask = ((self._timer.time > self.task_time_data['start_time']) &
                     (self._timer.time <= self.task_time_data['end_time']))

        real_attitude_BR = self.calculate_real_attitude_error()
        if hasattr(self, '_last_attitude_BR'):
            attitude_BR_dot = real_attitude_BR - self._last_attitude_BR
            attitude_BR_Binv = Binv(real_attitude_BR)
            angular_velocity_BR_B = 4 * torch.einsum(
                '...ij,...j->...i',
                attitude_BR_Binv,
                attitude_BR_dot,
            )
        else:
            angular_velocity_BR_B = torch.zeros_like(real_attitude_BR)

        self._last_attitude_BR = real_attitude_BR

        motion_error = torch.where(
            loss_mask.unsqueeze(-1),
            angular_velocity_BR_B,
            0.,
        )
        attitude_error = torch.where(
            loss_mask.unsqueeze(-1),
            real_attitude_BR,
            0.,
        )

        # sigma = e_hat * tan(\theta / 4)
        # in order to normalize the gradient
        # we multiply it by 16 as in 4**2

        attitude_loss = 8 * lp_loss(
            attitude_error,
            torch.zeros_like(attitude_error),
            reduction='sum',
        )
        motion_loss = torch.nn.functional.mse_loss(
            motion_error,
            torch.zeros_like(motion_error),
            reduction='sum',
        )

        true_torque: torch.Tensor = -self._simulator_state_dict['_spacecraft'][
            '_state_effectors']['_reaction_wheels']['current_torque'].detach()
        nominal_loss = torch.nn.functional.mse_loss(
            actions,
            true_torque.squeeze(),
        )

        self._timer.step()

        return (
            observation,
            Loss(
                attitude_loss=attitude_loss,
                battery_loss=battery_loss,
                loss_mask=loss_mask,
                nominal_loss=nominal_loss,
                motion_loss=motion_loss,
            ),
        )

    def state_dict(
            self) -> tuple[
                BuildConfig,
                LocationPointingConstellationStateDict,
            ]:
        return self.build_config, self._simulator_state_dict

    @abstractmethod
    def _load_taskset_config(
        self,
        taskset_config: list[TaskConfig],
    ) -> None:
        raise NotImplementedError

    def load_build_config(
        self,
        build_config: BuildConfig,
    ) -> None:
        orbits_data = build_config.get('orbits')
        self._orbits_data = (OrbitalElements.from_dicts(orbits_data)
                             if orbits_data else OrbitalElements.sample(
                                 self._num_envs))

        constellation_data = build_config.get('constellation')
        self._constellations_data = (
            Constellation.from_dicts(constellation_data)
            if constellation_data else Constellation.sample(self._num_envs))

        taskset_data = build_config.get('tasks')
        self._load_taskset_config(taskset_data)
        self._build_simulator()

        if self._num_envs != self._simulator.num_satellite:
            raise ValueError(
                'The number of envs does not match the constellation config.')

    def load_state_dict(
        self,
        build_config: BuildConfig,
        simulator_state_dict: BaseConstellationStateDict
        | None = None) -> None:
        self.load_build_config(build_config)
        self._simulator_state_dict = simulator_state_dict or self._simulator.reset(
        )

    def to(self, device: torch.device) -> None:
        self._simulator.to(device)
        self._simulator_state_dict = move_to(
            self._simulator_state_dict,
            device,
        )
