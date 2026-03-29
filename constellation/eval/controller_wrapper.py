__all__ = [
    'ControllerWrapper',
]
import pathlib

import einops
import pandas as pd
import todd
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
from ..loggers import EvalLogger
from ..task_managers import TaskManager

COMPLETION_RATE_THRESHOLD = 0.01


class ControllerWrapper:

    def __init__(
        self,
        split: str,
        rank: int,
        world_size: int,
        retry_from: pathlib.Path | None = None,
        gen_trajectory_dir: pathlib.Path | None = None,
    ) -> None:
        """ Eval Controller Wrapper
        Args:
            split(str): The dataset split to use, e.g., 'test'.
            rank(int): The rank of the index of the current environment..
            world_size(int): The total number of parallel environments in distributed evaluation.
            retry_from(pathlib.Path): If provided, only evaluate the instances whose completion rate is below the threshold in the given CSV file.
            gen_trajectory_dir(str): If provided, save the generated trajectories to this directory.
        """

        self._split = split

        # Here rank and world_size is not the same as RANK and WORLD_SIZE in distributed training.
        # here world_size refers to the number of parallel environments
        # and rank refers to the index of the current environment.
        self._rank = rank
        self._world_size = world_size

        self._annotations: list[int] = json_load(
            str(ANNOTATIONS_ROOT / f'{split}.json'),
        )['ids']
        if rank == 0:
            todd.logger.info(
                f"Total number of environment to evaluate: {len(self._annotations)}"
            )

        self._episode_step = 0
        self._controller: Controller | None = None
        self._last_num_succeeded_tasks = 0
        self._counter = -1
        self._device = 'cuda' if todd.Store.cuda else 'cpu'

        self._gen_trajectory_dir = gen_trajectory_dir

        if retry_from is not None:
            df = pd.read_csv(
                retry_from,
                names=['id', 'completion_rate'],
                index_col='id',
            )
            completion_rates: dict[int, float] = \
                df['completion_rate'].to_dict()
            self._annotations = [
                annotation for annotation in self._annotations if
                completion_rates.get(annotation, 0) < COMPLETION_RATE_THRESHOLD
            ]

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def _index(self) -> int:
        return self._counter * self._world_size + self._rank

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
        if self.all_done:
            return None
        if self.terminated or self.truncated:
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
            constellation_sensor_type=constellation_sensor_type - 1,
            constellation_sensor_enabled=constellation_sensor_enabled,
            constellation_data=constellation_data,
            tasks_sensor_type=sensor_type - 1,
            tasks_data=tasks_data,
        )
        return observation

    def _take_actions(self, task_indices: torch.Tensor) -> None:
        _controller = self._require_controller()

        task_indices = task_indices[:_controller.environment.num_satellites]
        _controller.memo['ongoing_tasks'
                         ] = _controller.task_manager.ongoing_tasks
        # FIXME: ControllerWrapper can never be used for annatation
        _controller.memo['assignment'] = task_indices

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
            device=self._device,
        )

        while _controller.task_manager.is_idle:
            self._take_actions(idle_action)

    def _get_annotation(self) -> int:
        return self._annotations[self._index]

    def reset(self) -> None:
        if self._controller is not None:
            self._controller.callbacks.after_run()
            self._controller = None

        self._counter += 1

        if self.all_done:
            return

        id_ = self._get_annotation()
        while (
            self._gen_trajectory_dir / f'{id_ // 1000:02d}' / f'{id_:05d}.json'
        ).exists():
            todd.logger.info(f"{id_} task has been evaluated, skiping")
            self._counter += 1
            if self.all_done:
                return
            id_ = self._get_annotation()

        save_dir = self._gen_trajectory_dir / f'{id_ // 1000:02d}'
        save_dir.mkdir(parents=True, exist_ok=True)

        self._episode_step = 0
        id_ = self._get_annotation()

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

        evalcallbacks = [
            CompletionRateEvaluator(),
            TurnAroundTimeEvaluator(),
            PowerUsageEvaluator(),
            EvalLogger(work_dir=self._gen_trajectory_dir),
        ]
        callbacks = ComposedCallback(callbacks=evalcallbacks)

        self._controller = Controller(
            name='test',
            environment=simulator,
            task_manager=task_manager,
            callbacks=callbacks,
        )
        self.memo['current_id'] = id_

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
