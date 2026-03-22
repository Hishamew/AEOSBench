__all__ = [
    "EvalRunner",
]

from typing import TYPE_CHECKING

import torch
from todd.runners import Memo
from tqdm import trange

from .constants import MAX_TIME_STEP
from .data.actions import Actions
from .eval import VecAlgorithm, VecController

if TYPE_CHECKING:
    from .callbacks import EvalRunnerComposedCallback


class EvalRunner:

    def __init__(
        self,
        name: str,
        *args,
        vec_controller: VecController,
        callbacks: 'EvalRunnerComposedCallback',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._name = name
        self._vec_controller = vec_controller

        self._callbacks = callbacks
        self._memo: Memo = dict()

        callbacks.bind(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def vec_controller(self) -> VecController:
        return self._vec_controller

    @property
    def callbacks(self) -> 'EvalRunnerComposedCallback':
        return self._callbacks

    def step(self, task_indices: torch.Tensor) -> None:
        self._callbacks.before_step()
        self.vec_controller.step(task_indices)
        self._callbacks.after_step()

    def run(
        self,
        algorithm: VecAlgorithm,
        *,
        max_time_step: int = MAX_TIME_STEP,
        progress_bar: bool = True,
    ) -> None:
        self._memo['algorithm'] = algorithm
        self._callbacks.before_run()

        for _ in trange(max_time_step, disable=not progress_bar):
            if self._callbacks.should_break():
                break

            task_indices = algorithm.step(
                self.vec_controller.get_observations()
            )
            self.step(task_indices)

        self._callbacks.after_run()
