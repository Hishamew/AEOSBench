__all__ = [
    'AttitudeControlEnvironment',
]
from typing import Any

import spiceypy
import todd
import torch
from satsim.architecture import Timer
from satsim.simulation.gravity import Ephemeris
from satsim.utils import dict_recursive_apply
from todd.utils import StateDictMixin

from constellation.constants import TIMESTAMP
from constellation.data import Constellation

from ..constants import INITIAL_EPHEMERIS_PATH
from ..taskset import LocationPointingTaskset as TaskSet
from .constellation import AttitudeControlConstellation as Simulator
from .constellation import (
    AttitudeControlConstellationStateDict as SimulatorStateDict,
)


class AttitudeControlEnvironment(StateDictMixin):

    def __init__(
        self,
        num_envs: int,
        backend: torch.device | None = None,
        fp_precision: torch.dtype = torch.float64,
    ) -> None:
        self._num_envs = num_envs
        self._initial_ephemeris: Ephemeris = torch.load(
            str(INITIAL_EPHEMERIS_PATH)
        )
        if backend is not None:
            self._backend = backend
        else:
            self._backend = torch.device('cuda' if todd.Store.cuda else 'cpu')
        self._fp_precision = fp_precision

    @property
    def backend(self) -> torch.device:
        return self._backend

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @num_envs.setter
    def num_envs(self, num_envs: int) -> None:
        self._num_envs = num_envs

    @property
    def is_built(self) -> bool:
        return hasattr(self, "_simulator")

    @property
    def ephemeris(self) -> Ephemeris:
        if not self.is_built:
            return self._initial_ephemeris
        else:
            return self._simulator.get_earth_ephemeris(None)

    @property
    def simulator(self) -> Simulator:
        if not self.is_built:
            raise AttributeError("Constellation not initialized yet.")
        return self._simulator

    @property
    def simulator_state_dict(self) -> SimulatorStateDict:
        if not self.is_built:
            raise AttributeError("Constellation not initialized yet.")
        return self._simulator_state_dict

    @property
    def constellation(self) -> Constellation:
        return self._constellation

    @property
    def taskset(self) -> TaskSet:
        return self._taskset

    @constellation.setter
    def constellation(self, constellation: Constellation) -> None:
        self._constellation = constellation
        self.num_envs = len(constellation)

    @taskset.setter
    def taskset(self, taskset: TaskSet) -> None:
        self._taskset = taskset

    def build_simulator(self) -> None:
        spiceypy.kclear()
        self._timer = Timer(1.)
        self._simulator = Simulator(
            self._timer,
            self._constellation,
            TIMESTAMP,
            self._taskset,
        )
        self._simulator_state_dict = self._simulator.reset()
        self._timer.reset()

        self._simulator.to(self._backend, self._fp_precision)
        self._simulator_state_dict = dict_recursive_apply(
            self._simulator_state_dict,
            lambda x: x.to(self._backend, self._fp_precision),
        )
        self.setup_tracking_target()

    def setup_tracking_target(self) -> None:
        lla = self._simulator.tracking_target.new_tensor([
            task.coordinate for task in self._taskset
        ])
        with_target = torch.ones_like(self._simulator.with_target)

        self._simulator.take_actions(
            ~self._simulator.camera_switch,
            lla,
            with_target,
        )

    def step(self) -> None:
        self._simulator_state_dict, _ = self._simulator(
            self._simulator_state_dict
        )
        self._timer.step()

    def clear_grad(self) -> None:
        self._simulator_state_dict = dict_recursive_apply(
            self._simulator_state_dict,
            lambda x: x.clone().detach(),
        )

    def state_dict(self) -> dict[str, Any]:
        if not self.is_built:
            raise AttributeError("Constellation not initialized yet.")

        return dict(
            constellation=self._constellation.to_dict(),
            taskset=self._taskset.to_dicts(),
            simulator_state_dict=self._simulator_state_dict,
        )

    def load_state_dict(self, state_dict, *args, **kwargs):
        if 'constellation' in state_dict:
            self._constellation = Constellation.from_dict(
                state_dict['constellation']
            )

        if 'taskset' in state_dict:
            self._taskset = TaskSet.from_dicts(state_dict['taskset'])

        if 'simulator_state_dict' in state_dict:
            self._simulator_state_dict = state_dict['simulator_state_dict']
            self._simulator_state_dict = dict_recursive_apply(
                self._simulator_state_dict,
                lambda x: x.to(self._backend, self._fp_precision),
            )
