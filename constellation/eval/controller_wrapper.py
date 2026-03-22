__all__ = [
    'ControllerWrapper',
]
import random
from typing import Any

import einops
import torch
from todd.patches.py_ import json_load
from todd.runners import Memo

from ..callbacks import ComposedCallback
from ..constants import (
    ANNOTATIONS_ROOT,
    CONSTELLATIONS_ROOT,
    TASKSETS_ROOT,
    TIMESTAMP,
)
from ..controller import Controller
from ..data import Constellation, TaskSet
from ..environments import SatsimEnvironment
from ..eval import Observation
from ..evaluators import (
    CompletionRateEvaluator,
    PowerUsageEvaluator,
    TurnAroundTimeEvaluator,
)
from ..task_managers import TaskManager


class ControllerWrapper:

    def __init__(self, split: str) -> None:
        self._split = split

        self._annotations: list[int] = json_load(
            str(ANNOTATIONS_ROOT / f'{split}.json'),
        )['ids']

        self._episode_step = 0
        self._controller: Controller | None = None
        self._last_num_succeeded_tasks = 0

    def _require_controller(self) -> Controller:
        if self._controller is None:
            raise RuntimeError("Controller is not initialized")
        return self._controller

    @property
    def terminated(self) -> bool:
        _controller = self._require_controller()
        return _controller.task_manager.all_closed

    @property
    def truncated(self) -> bool:
        _controller = self._require_controller()
        return _controller.environment.timer.time >= 3600

    @property
    def all_done(self) -> bool:
        return self._index >= len(self._annotations)

    @property
    def memo(self) -> Memo:
        _controller = self._require_controller()
        return _controller.memo

    def get_observation(self) -> Observation | None:
        if self.terminated or self.truncated or self.all_done:
            return None

        _controller = self._require_controller()
        env = _controller.environment
        (
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
        ) = env.get_observation()
        time_step = env.timer.time

        task_manager = _controller.task_manager

        valid_tasks = task_manager.ongoing_tasks
        valid_labels = task_manager.ongoing_flags

        sensor_type, static_data = valid_tasks.to_tensor()

        static_data = static_data.clone()
        static_data[..., 0] -= time_step
        static_data[..., 1] -= time_step

        progress = task_manager.progress[valid_labels]
        dynamic_data = einops.rearrange(progress, 'nt -> nt 1')
        tasks_data = torch.cat([static_data, dynamic_data], -1)
        observation = Observation(
            num_satellites=env.num_satellites,
            num_tasks=task_manager.num_ongoing_tasks,
            time_step=time_step,
            constellation_sensor_type=constellation_sensor_type,
            constellation_sensor_enabled=constellation_sensor_enabled,
            constellation_data=constellation_data,
            tasks_sensor_type=sensor_type - 1,
            tasks_data=tasks_data,
        )
        return observation

    def _take_actions(self, task_indices: torch.Tensor) -> None:
        _controller = self._require_controller()

        task_indices = task_indices[:_controller.environment.num_satellites]
        _controller.memo['task_indices'] = task_indices
        _controller.memo['ongoing_tasks'
                         ] = _controller.task_manager.ongoing_tasks

        _controller.callbacks.before_step()

        is_visible = _controller.environment.is_visible(
            _controller.task_manager.taskset
        )
        _controller.memo['is_visible'] = is_visible
        _controller.task_manager.record(is_visible)

        _controller.environment.take_actions_with_tensor(
            task_indices=task_indices,
            tasks=_controller.task_manager.ongoing_tasks,
        )

        _controller.callbacks.after_step()

        _controller.environment.timer.step()
        _controller.environment.step()

    def _skip_idle(self) -> None:
        _controller = self._require_controller()
        idle_action = torch.full(
            (_controller.environment.num_satellites, ),
            -1,
        )

        while _controller.task_manager.is_idle:
            self._take_actions(idle_action)

    def _get_annotation(self) -> int:
        return random.choice(self._annotations['ids'])

    def reset(self) -> None:
        if self.all_done:
            return

        if self._controller is not None:
            self._controller.callbacks.after_run()

        self._episode_step = 0
        id_ = self._get_annotation()
        self.memo['current_id'] = id_

        constellation_path = (
            CONSTELLATIONS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        constellation = Constellation.load(str(constellation_path))

        taskset_path = (
            TASKSETS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        tasks: TaskSet = TaskSet.load(str(taskset_path))

        simulator = SatsimEnvironment(
            standard_time_init=TIMESTAMP,
            constellation=constellation,
            all_tasks=tasks,
        )

        task_manager = TaskManager(
            timer=simulator.timer,
            taskset=tasks,
        )
        self._last_num_succeeded_tasks = 0

        evaluators = [
            CompletionRateEvaluator(),
            TurnAroundTimeEvaluator(),
            PowerUsageEvaluator(),
        ]
        callbacks = ComposedCallback(callbacks=evaluators)

        self._controller = Controller(
            name='test',
            environment=simulator,
            task_manager=task_manager,
            callbacks=callbacks,
        )

        callbacks.before_run()
        self._skip_idle()

    def step(
        self,
        task_indices: torch.Tensor,
    ) -> None:
        if self.all_done:
            return
        _controller = self._require_controller()
        self._take_actions(task_indices)
        self._last_num_succeeded_tasks = _controller.task_manager.num_succeeded_tasks
        if not (self.terminated or self.truncated):
            self._skip_idle()
