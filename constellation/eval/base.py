__all__ = [
    'Observation',
    'VecEnv',
    'VecAlgorithm',
]
from abc import ABC, abstractmethod
from typing import TypedDict

import torch
from todd.runners import Memo
from torch import nn


class Observation(TypedDict):
    num_satellites: int
    num_tasks: int
    time_step: int
    constellation_sensor_type: torch.Tensor
    constellation_sensor_enabled: torch.Tensor
    constellation_data: torch.Tensor
    tasks_sensor_type: torch.Tensor
    tasks_data: torch.Tensor


class VecEnv(ABC):

    def __init__(self, num_controllers: int) -> None:
        self._num_controllers = num_controllers

    @property
    @abstractmethod
    def controllers_memo(self) -> list[Memo]:
        raise NotImplementedError

    @property
    @abstractmethod
    def all_done(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        task_indices: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_observations(self) -> list[Observation]:
        raise NotImplementedError


class VecAlgorithm(nn.Module, ABC):

    @abstractmethod
    def step(
        self,
        observations: list[Observation],
    ) -> torch.Tensor:
        pass
