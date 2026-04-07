__all__ = [
    'ControllerRunner',
]
import logging
import pathlib
from collections import defaultdict
from typing import TYPE_CHECKING

import todd
import torch
from todd.configs import PyConfig
from todd.runners import Memo
from torch import nn
from tqdm import trange

from .environment.environment import AttitudeControlEnvironment
from .model import MLPPIDConfigure

if TYPE_CHECKING:
    from .callbacks import ComposedCallback


class ControllerRunner:

    def __init__(
        self,
        model: nn.Module,
        callbacks: 'ComposedCallback',
        config: PyConfig,
        env: AttitudeControlEnvironment,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self._model = model
        self._optim = optimizer
        self._env = env
        self._memo: Memo = defaultdict(dict)

        self._memo['work_dir'] = pathlib.Path(config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        callbacks.bind(self)
        self._callbacks = callbacks

        self._memo['config'] = config

    @property
    def config(self) -> PyConfig:
        return self._memo['config']

    @property
    def full_grad(self) -> bool:
        return self.config.get('full_grad', False)

    @property
    def total_episode(self) -> int:
        return self.config.total_episode

    @property
    def episode_length(self) -> int:
        return self.config.episode_length

    @property
    def work_dir(self) -> pathlib.Path:
        return self._memo['work_dir']

    @property
    def memo(self) -> Memo:
        return self._memo

    @property
    def logger(self) -> logging.Logger:
        return todd.logger

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        return self._optim

    @property
    def environment(self) -> AttitudeControlEnvironment:
        return self._env

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def episode(self) -> int:
        return self._memo['episode']

    @property
    def tag(self) -> str:
        return self._memo['tag']

    @episode.setter
    def episode(self, value: int) -> None:
        self._memo['episode'] = value
        self.tag = f'episode_{value}'

    @tag.setter
    def tag(self, value: str) -> None:
        self._memo['tag'] = value

    def run_episode(self) -> None:

        self._callbacks.before_episode()

        for step in trange(
            self.episode_length, disable=not self.config.progress_bar
        ):
            self._callbacks.before_step()
            self.environment.step()
            self._callbacks.after_step()

        self._callbacks.after_episode()

    def run(self) -> None:

        self._callbacks.before_run()

        for episode in range(self.total_episode):
            self.episode = episode
            self.run_episode()

        self.tag = 'after_run'
        self._callbacks.after_run()

        self.logger.info("Running completed.")
