__all__ = [
    'GammaScheduler',
    'ControllerRunner',
]
import logging
import os
import random
from itertools import product
from pprint import pformat
from typing import Any, Protocol

import torch
from todd.configs import PyConfig
from todd.loggers import Formatter, master_logger
from todd.patches.py_ import json_dump, json_load, run
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .base import BaseEvaluator, BaseModel
from .benchmark import Benchmark
from .environment import AttitudeControlEnvironment, BuildConfig
from .monitor import BaseStateMonitor
from .registry import EnvironmentRegistry, MonitorRegistry


class GammaScheduler:

    def __init__(
        self,
        gamma_init: float,
        total_step: int,
        gamma_end: float = 1.,
    ):
        self._gamma_init = gamma_init
        self._gamma_end = gamma_end
        self._episode_length = total_step
        self._current_step = 0

    def step(self) -> float:
        if self._current_step >= self._episode_length:
            return self._gamma_end

        gamma_t = self._gamma_init + (self._gamma_end - self._gamma_init) * (
            self._current_step / self._episode_length)
        self._current_step += 1
        return gamma_t


class Optimizer(Protocol):

    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        pass


class ControllerRunner:

    def __init__(
        self,
        model: BaseModel,
        config: PyConfig,
        benchmark: Benchmark,
        env: AttitudeControlEnvironment,
        optimizer: Optimizer | None = None,
    ):
        self._model = model
        self._optim = optimizer
        self._env = env
        self._benchmark = benchmark

        self._config = config
        master_logger.info(f'Training Config:\n {pformat(config)}')

        self._log_dir: str = config.train.log_dir
        self._model_save_dir = os.path.join(self._log_dir, 'checkpoints')
        self._save_period = config.train.get('save_period', 10 * 360)
        os.makedirs(self._model_save_dir, exist_ok=True)
        config.dump(os.path.join(config.train.log_dir, 'train_config.py'))

        git_commit_hash = run('git rev-parse HEAD')
        with open(
                os.path.join(config.train.log_dir, 'git_commit_hash.txt'),
                'w',
                encoding='utf8',
        ) as f:
            f.write(git_commit_hash)

        if config.train.get('file_logging', False):
            file = os.path.join(config.train.log_dir, 'train.log')
            handler = logging.FileHandler(file)
            handler.setFormatter(Formatter())
            master_logger.addHandler(handler)

        config = config.train

        self._total_step: int = config.total_step
        self._rollout_length: int = config.rollout_length
        self._episode_length: int = config.episode_length
        self._loss_weight = config.loss
        self._clip_grad_norm = config.get('clip_grad_norm', None)

        self._task_sampler = config.get('task_sampler', dict())

        self._gamma_scheduler = GammaScheduler(
            config.gamma,
            config.gamma_step,
        )

        default_monitor_configs = [
            dict(type='AttitudeErrorMonitor'),
            dict(type='BatteryMonitor'),
            dict(type='TorqueMonitor'),
        ]
        monitors_config = config.get(
            'monitor',
            [],
        )
        default_monitor_configs.extend(monitors_config)
        self._monitors: list[BaseStateMonitor] = [
            MonitorRegistry.build(config) for config in default_monitor_configs
        ]

        self._tensorboard_dir = os.path.join(
            config.log_dir,
            'tensorboard',
        )

    def load_checkpoints(self, path: str) -> None:
        env_build_config = json_load(
            os.path.join(path, 'env_build_config.json'))
        simulator_state_dict = torch.load(
            os.path.join(path, 'simulator_state_dict.pth'))
        model_state_dict = torch.load(
            os.path.join(path, 'model_checkpoints.pth'))

        self._model.load_state_dict(model_state_dict)
        self._env.load_state_dict(
            env_build_config,
            simulator_state_dict,
        )

    def save_checkpoints(self, dir_name: str) -> None:
        path = dir_name
        os.makedirs(path, exist_ok=True)

        (
            env_build_config,
            simulator_state_dict,
        ) = self._env.state_dict()
        model_state_dict = self._model.state_dict()

        json_dump(
            env_build_config,
            os.path.join(path, 'env_build_config.json'),
            indent=4,
        )
        torch.save(
            simulator_state_dict,
            os.path.join(path, 'simulator_state_dict.pth'),
        )
        torch.save(
            model_state_dict,
            os.path.join(path, 'model_checkpoints.pth'),
        )

    def train(
        self,
        resume_from: str | None = None,
    ) -> None:
        if not self._optim:
            raise RuntimeError(
                "Trying to train a controller without providing a optimizer")

        tensorboard = SummaryWriter(self._tensorboard_dir)
        if resume_from is not None:
            master_logger.info("Checkpoint is given, using checkpoint")
            self.load_checkpoints(resume_from)
            observation, _ = self._env.reset(**self._task_sampler)
        elif self._env.has_initialized:
            master_logger.info(
                "Training for special case, make sure your sampler "
                "doesn't erase your pre-defined config")
            master_logger.info(f"Sampler:\n {pformat(self._task_sampler)}")
            observation, _ = self._env.reset(**self._task_sampler, )
        else:
            master_logger.info("No pre-defined config is loaded. "
                               "Initializing environment by random sample.")
            observation, _ = self._env.reset()

        if self._config.train.evaluate_initial_model:
            master_logger.info("Before Training Performance")
            self.test('before_training')
            benchmark_results = self.benchmark('before_training')

        gamma = self._gamma_scheduler.step()

        rollout_attitude_loss = 0.
        average_non_decayed_loss = 0.
        total_loss_count = 0.
        rollout_nominal_loss = 0.
        rollout_motion_loss = 0.
        for step_idx in range(self._total_step):

            actions = self._model(observation)
            observation, losses = self._env.step(actions)

            loss_mask = losses['loss_mask']
            loss_count = loss_mask.sum().item()
            attitude_loss = losses['attitude_loss']
            rollout_attitude_loss = gamma * rollout_attitude_loss + attitude_loss
            total_loss_count += loss_count

            rollout_nominal_loss = rollout_nominal_loss + losses['nominal_loss']

            rollout_motion_loss = rollout_motion_loss + losses['motion_loss']
            with torch.no_grad():
                average_non_decayed_loss += attitude_loss
            if (step_idx + 1) % self._rollout_length == 0:
                total_loss_count = max(total_loss_count, 1)
                rollout_attitude_loss = (
                    rollout_attitude_loss) / total_loss_count

                battery_loss = (losses['battery_loss'])
                display_battery_loss = battery_loss.item()

                rollout_nominal_loss = (
                    rollout_nominal_loss) / self._rollout_length
                display_nominal_loss = rollout_nominal_loss.item()

                rollout_motion_loss = (rollout_motion_loss) / total_loss_count
                display_motion_loss = rollout_motion_loss.item()

                motion_loss_weight = self._loss_weight.get('motion_loss', 0.)
                rollout_loss = (
                    rollout_attitude_loss * self._loss_weight.attitude_loss +
                    battery_loss * self._loss_weight.battery_loss +
                    rollout_nominal_loss * self._loss_weight.nominal_loss +
                    rollout_motion_loss * motion_loss_weight)
                display_rollout_loss = (rollout_attitude_loss + battery_loss +
                                        rollout_nominal_loss +
                                        rollout_motion_loss).item()

                average_non_decayed_loss = (average_non_decayed_loss /
                                            total_loss_count)
                master_logger.info(
                    f"Step: {step_idx+1}/ {self._total_step} Gamma: {gamma:.8f} "
                    f"Training loss: {display_rollout_loss:.8f}\n"
                    f"Integral Average Error: {average_non_decayed_loss:.8f} "
                    f"Energy Cost: {losses['battery_loss']:.8f} "
                    f"Nominal loss: {display_nominal_loss:.8f} "
                    f"Motion loss: {display_motion_loss:.8f}")

                tensorboard.add_scalar(
                    "Train/Training Loss",
                    display_rollout_loss,
                    step_idx + 1,
                )
                tensorboard.add_scalar(
                    "Train/Integral Average Attitude Error",
                    average_non_decayed_loss,
                    step_idx + 1,
                )
                tensorboard.add_scalar(
                    "Train/Energy Cost",
                    display_battery_loss,
                    step_idx + 1,
                )
                tensorboard.add_scalar(
                    "Train/Nominal Loss",
                    display_nominal_loss,
                    step_idx + 1,
                )
                tensorboard.add_scalar(
                    "Train/Motion Loss",
                    display_motion_loss,
                    step_idx + 1,
                )

                self._optim.zero_grad()
                rollout_loss.backward()
                for name, param in self._model.named_parameters():
                    if param.requires_grad == False or param.grad is None:
                        continue
                    if torch.isnan(param.grad).any():
                        raise RuntimeError(
                            f"NaN grad detected, deported. Params Name: {name}"
                        )

                if self._clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        self._clip_grad_norm,
                    )
                self._optim.step()

                rollout_attitude_loss = 0.
                average_non_decayed_loss = 0.
                total_loss_count = 0.
                rollout_nominal_loss = 0.
                rollout_motion_loss = 0.

                gamma = self._gamma_scheduler.step()

                # Cut gradient from last rollout
                self._env.clear_grad()

                observation = {
                    k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                    for k, v in observation.items()
                }
                losses = dict(
                    attitude_loss=0.,
                    battery_loss=0.,
                    nominal_loss=0.,
                    motion_loss=0.,
                )

            if (step_idx + 1) % self._episode_length == 0:
                observation, _ = self._env.reset(**self._task_sampler, )

            if (step_idx + 1) % self._save_period == 0:
                self.save_checkpoints(
                    os.path.join(self._model_save_dir,
                                 f'checkpoints_{step_idx+1}'))
                self.test(f'step_{step_idx + 1}')
                benchmark_results = self.benchmark(f'step_{step_idx + 1}')
                for key, value in benchmark_results.items():
                    tensorboard.add_scalar(
                        f'Benchmark/{key}',
                        value,
                        step_idx + 1,
                    )

        self.save_checkpoints(
            os.path.join(self._model_save_dir, f'checkpoints_{step_idx+1}'))
        self.test(f'step_{step_idx + 1}')
        master_logger.info("Training completed.")

        tensorboard.close()

    @torch.no_grad()
    def test(self, tag: str) -> None:
        num_envs = 9
        env_kwargs = dict(self._config.environment)
        env_kwargs.update(
            num_envs=9,
            episode_length=self._episode_length,
        )
        test_env: AttitudeControlEnvironment = EnvironmentRegistry.build(
            env_kwargs)
        env_ids = list(range(num_envs))

        if not self._env.has_initialized:
            self._env.reset()
        build_config = self._env.build_config
        env_ids = random.sample(range(self._env.num_envs), num_envs)
        build_config = {
            'orbits': [build_config['orbits'][env_id] for env_id in env_ids],
            'constellation':
            [build_config['constellation'][env_id] for env_id in env_ids],
            'tasks': [build_config['tasks'][env_id] for env_id in env_ids],
        }

        test_env.load_build_config(build_config)

        observation, _ = test_env.reset(**self._task_sampler, )
        task_time = test_env.task_time_data
        start_times = task_time['start_time'].cpu().tolist()
        end_times = task_time['end_time'].cpu().tolist()

        for episode_step in range(self._episode_length):
            actions: torch.Tensor = self._model(
                observation,
                deterministic=True,
                env_ids=env_ids,
            )
            observation, _ = test_env.step(actions)

            for monitor in self._monitors:
                monitor(
                    test_env,
                    observation,
                    test_env.simulator_state_dict,
                )

        test_save_dir = os.path.join(self._log_dir, 'test', tag)
        os.makedirs(test_save_dir, exist_ok=True)

        for monitor in self._monitors:
            monitor.plot(
                test_save_dir,
                start_times,
                end_times,
            )

    def to_device(self, device: torch.device) -> None:
        self._model.to(device)

    @torch.no_grad()
    def benchmark(
        self,
        tag: str | None = None,
    ) -> dict[str, float]:

        master_logger.info(f"Benchmark Results over test split tasks:")
        if tag:
            os.makedirs(
                os.path.join(self._log_dir, 'benchmark'),
                exist_ok=True,
            )
            benchmark_save_path = os.path.join(
                self._log_dir,
                'benchmark',
                f'{tag}.txt',
            )
            with open(benchmark_save_path, 'w', encoding='utf8') as f:
                f.write("Benchmark Results over test split tasks:\n")
        else:
            benchmark_save_path = None
        results = self._benchmark.evaluate(self._model, benchmark_save_path)

        return results
