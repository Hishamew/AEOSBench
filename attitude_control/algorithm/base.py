__all__ = [
    'BaseStateMonitor',
    'BaseModel',
    'BaseEvaluator',
]
import os
from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import torch
from todd.loggers import master_logger
from torch import nn

from .constellation import BaseConstellationStateDict
from .constellation.base_constellation import BaseConstellation
from .environment import Loss, Observation
from .environment.base_environment import AttitudeControlEnvironment
from .registry import MonitorRegistry


class BaseStateMonitor(ABC):

    def __init__(
        self,
        state_name: str,
        state_unit: str,
        num: int = 9,
    ) -> None:
        self._state_list = [[] for _ in range(num)]
        self._state_name = state_name
        self._state_unit = state_unit
        self._num = num

    @abstractmethod
    def pick_state(
        self,
        environment: AttitudeControlEnvironment,
        observation: Observation,
        state_dict: BaseConstellationStateDict,
    ) -> torch.Tensor:
        pass

    def __call__(
        self,
        environment: AttitudeControlEnvironment,
        observation: Observation,
        state_dict: BaseConstellationStateDict,
    ) -> None:
        state = self.pick_state(environment, observation, state_dict)

        state = self._post_hook(state)

        for i in range(self._num):
            self._state_list[i].append(state[i])

    def _post_hook(self, state: torch.Tensor) -> torch.Tensor:
        return state.cpu().tolist()

    def plot(
        self,
        save_path: str,
        start_time: list[float] | None = None,
        end_time: list[float] | None = None,
    ) -> None:
        plt.figure(figsize=(15, 15))
        for i in range(self._num):
            plt.subplot(3, 3, i + 1)
            plt.plot(self._state_list[i])
            if start_time:
                plt.axvline(
                    x=start_time[i],
                    linestyle='--',
                    color='green',
                    label='Start',
                )
            if end_time:
                plt.axvline(
                    x=end_time[i],
                    linestyle='--',
                    color='red',
                    label='End',
                )

            plt.xlabel('Timestep')
            plt.ylabel(f'{self._state_name} ({self._state_unit})')
            plt.title(f'Env {i+1} {self._state_name}')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{self._state_name}.png'))
        plt.close()

        self._state_list = [[] for i in range(self._num)]


class BaseModel(nn.Module, ABC):

    @abstractmethod
    def forward(
        self,
        observation: Observation,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        pass


class BaseEvaluator(ABC):

    def __init__(self):
        super().__init__()
        self._init_state()

    @abstractmethod
    def __call__(
        self,
        environment: AttitudeControlEnvironment,
        state_dict: BaseConstellationStateDict,
        loss_info: Loss,
    ) -> None:
        pass

    @abstractmethod
    def _evaluate(self) -> tuple[dict[str, Any], str]:
        pass

    def _init_state(self) -> None:
        pass

    def evaluate(self, save_path: str | None = None) -> dict[str, Any]:
        result, log = self._evaluate()
        master_logger.info(log)

        if save_path:
            with open(save_path, 'a+', encoding='utf8') as f:
                f.write(log + '\n')

        self._init_state()
        return result
