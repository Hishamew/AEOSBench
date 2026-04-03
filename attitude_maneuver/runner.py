__all__ = [
    'ControllerRunner',
]
import logging
import pathlib
from collections import defaultdict
from pprint import pformat
from typing import TYPE_CHECKING

import todd
import torch
from todd.configs import PyConfig
from todd.runners import Memo
from tqdm import trange

from .environment.environment import AttitudeControlEnvironment
from .model import MLPPIDConfigure

if TYPE_CHECKING:
    from .callbacks import ComposedCallback


class ControllerRunner:

    def __init__(
        self,
        model: MLPPIDConfigure,
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
    def model(self) -> MLPPIDConfigure:
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
        self.load_actuator()

        for step in trange(
            self.episode_length, disable=not self.config.progress_bar
        ):
            self._callbacks.before_step()
            self.environment.step()
            self._callbacks.after_step()

        self._callbacks.after_episode()

    def load_actuator(self) -> None:
        simulator = self.environment.simulator
        rw = simulator.reaction_wheels
        rw_inertia = rw.moment_of_inertia_wrt_spin.squeeze()

        hub = simulator.spacecraft.hub
        sc_mass = hub.mass
        sc_inertia = hub.moment_of_inertia_matrix_wrt_body_point
        sc_inertia = torch.diagonal(sc_inertia, dim1=-2, dim2=-1)

        params_dtype = torch.get_default_dtype()
        pid_params: torch.Tensor = self.model(
            sc_inertia=sc_inertia.to(params_dtype),
            rw_inertia=rw_inertia.to(params_dtype),
            sc_mass=sc_mass.to(params_dtype),
        )
        k, ki, p, integral_limit = pid_params.unbind(-1)
        self.environment.simulator.configure_pid(
            self.environment.simulator_state_dict,
            k,
            ki,
            p,
            integral_limit,
            self.full_grad,
        )

    def run(self) -> None:

        self._callbacks.before_run()

        for episode in range(self.total_episode):
            self.episode = episode
            self.run_episode()

        self.tag = 'after_run'
        self._callbacks.after_run()

        self.logger.info("Training completed.")
