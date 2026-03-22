__all__ = [
    'DummyVecControllerEnv',
]
from typing import Any, Callable, Iterable

import torch
from todd.runners import Memo

from .base import Observation, VecEnv
from .controller_wrapper import ControllerWrapper


class QueuedMemoBound:

    def __init__(self, instance: 'DummyVecControllerEnv'):
        self._instance = instance

    def __getitem__(self, key: str) -> object:
        values: list[Any] = []
        for controller in self._instance.controllers:
            values.append(controller.memo[key])
        return values

    def __setitem__(self, key: str, value: list[Any]) -> None:
        for controller, val in zip(
            self._instance.controllers,
            value,
            strict=True,
        ):
            controller.memo[key] = val


class QueuedMemo:

    def __get__(
        self,
        instance: 'DummyVecControllerEnv',
        owner,
    ) -> QueuedMemoBound:
        if instance is None:
            raise AttributeError('Cannot access QueuedMemo from class')
        return QueuedMemoBound(instance)


class DummyVecControllerEnv(VecEnv):
    _controllers_memo = QueuedMemo()

    def __init__(
        self,
        controllers_fn: Iterable[Callable[[], ControllerWrapper]],
    ) -> None:
        self._controllers = [fn() for fn in controllers_fn]
        super().__init__(num_controllers=len(self._controllers))

    @property
    def controllers(self) -> list[ControllerWrapper]:
        return self._controllers

    @property
    def controllers_memo(self) -> QueuedMemoBound:
        return self._controllers_memo

    @property
    def all_done(self) -> bool:
        return all(controller.all_done for controller in self.controllers)

    def step(
        self,
        task_indices: torch.Tensor,
    ) -> None:
        for controller, task_assignment in zip(
            self.controllers,
            task_indices.unbind(0),
        ):
            controller.step(task_assignment)
            if controller.terminated or controller.truncated:
                controller.reset()

    def get_observations(self) -> list[Observation]:
        observations = []
        for controller in self.controllers:
            observation = controller.get_observation()
            if observation is not None:
                observations.append(observation)

        return observations
