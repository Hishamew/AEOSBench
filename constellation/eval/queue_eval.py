__all__ = [
    'QueueController',
]
from typing import Any, Iterable

import einops
import torch
from todd.runners import Memo

from ..controller import Controller
from ..data.actions import Actions
from .base import Observation, VecController


class QueuedMemoBound:

    def __init__(self, instance: 'QueueController'):
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
        instance: 'QueueController',
        owner,
    ) -> QueuedMemoBound:
        if instance is None:
            raise AttributeError('Cannot access QueuedMemo from class')
        return QueuedMemoBound(instance)


class QueueController(VecController):
    _controllers_memo = QueuedMemo()

    def __init__(self, controllers: Iterable[Controller]):
        self._controllers = list(controllers)
        super().__init__(num_controllers=len(self._controllers))
        self._memo: Memo = dict()

    @property
    def controllers(self) -> list[Controller]:
        return self._controllers

    @property
    def memo(self) -> Memo:
        return self._memo

    @property
    def controllers_memo(self) -> QueuedMemoBound:
        return self._controllers_memo

    def step(
        self,
        task_indices: torch.Tensor,
    ) -> None:
        for controller, task_assignment in zip(
            self.controllers,
            task_indices.unbind(0),
        ):
            controller.step_with_task_indices(task_assignment)

    def get_observations(self) -> list[Observation]:
        observations = []
        for controller, static_observation in zip(
            self.controllers,
            self._memo['static_constellation_observations'],
        ):
            observation = controller.get_observation()
            observations.append(observation)

        return observations
