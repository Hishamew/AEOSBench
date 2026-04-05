import pathlib
import random
from abc import ABC, abstractmethod
from typing import Any

import torch
from matplotlib import pyplot as plt

from ..callbacks import BaseCallback


class BaseMonitor(BaseCallback, ABC):

    def __init__(
        self,
        *args,
        name: str,
        unit: str,
        interval: int = 20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._name = name
        self._unit = unit
        self._interval = interval

    def bind(self, *args, **kwargs):
        super().bind(*args, **kwargs)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def work_dir(self) -> pathlib.Path:
        return self.runner.memo['work_dir'] / 'monitor'

    @property
    def name(self) -> str:
        return self._name

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def tag(self) -> str:
        return self.runner.memo['tag']

    @property
    def num(self) -> int:
        return 9

    @property
    def recorder(self) -> None | list[Any]:
        return self.runner.memo['monitor'].get(self.name, None)

    @recorder.setter
    def recorder(self, value: list[Any]) -> None:
        self.runner.memo['monitor'][self.name] = value

    @property
    def should_record(self) -> bool:
        return self.runner.episode % self._interval == 0

    def before_episode(self) -> None:
        if self.should_record:
            self.recorder = []

    def after_episode(self) -> None:
        if self.should_record:
            data = torch.stack(self.recorder, dim=1)
            data = random.sample(data.tolist(), self.num)
            self.plot(data)

    def after_step(self):
        if not self.should_record:
            return

        self._should_after_step()

    @abstractmethod
    def _should_after_step(self) -> None:
        pass

    def plot(self, data: list[Any]) -> None:
        plt.figure(figsize=(15, 15))
        for i in range(self.num):
            plt.subplot(3, 3, i + 1)
            plt.plot(data[i])

            plt.xlabel('Timestep')
            plt.ylabel(f'{self.name} ({self.unit})')
            plt.title(f'Env {i+1} {self.name}')
            plt.legend()
        plt.tight_layout()
        (self.work_dir / self.tag).mkdir(parents=True, exist_ok=True)
        plt.savefig(str(self.work_dir / self.tag / f'{self._name}.png'))
        plt.close()
