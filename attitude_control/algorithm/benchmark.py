import os
from typing import Type

import torch
from matplotlib.font_manager import json_load

from .base import BaseEvaluator, BaseModel
from .constellation.pertubation import NoisePool
from .environment import (AttitudeControlEnvironment,
                          AttitudeMaintainEnvironment, BuildConfig,
                          LocationPointingEnvironment,
                          SpacecraftPointingEnvironment)
from .registry import BenchmarkRegistry, EvaluatorRegistry


class Benchmark:

    def __init__(
        self,
        evaluators: list[BaseEvaluator],
        env_builder: Type[AttitudeControlEnvironment],
        build_config: BuildConfig,
        **kwargs,
    ) -> None:
        self._episode_length = 360
        self._build_config = build_config
        self._env_builder = env_builder
        self._evaluators = evaluators
        self._additional_kwargs = kwargs

    @torch.no_grad()
    def evaluate(
        self,
        model: BaseModel,
        save_path: str | None,
    ) -> dict[str, float]:
        env = self._env_builder(
            num_envs=len(self._build_config['constellation']),
            build_config=self._build_config,
            **self._additional_kwargs,
        )
        observation, _ = env.reset(
            sample_constellation=False,
            sample_task=False,
        )

        for episode_step in range(self._episode_length):
            actions = model(
                observation,
                deterministic=True,
            )
            observation, loss_info = env.step(actions)
            for evaluator in self._evaluators:
                evaluator(env, env.simulator_state_dict, loss_info)

        if save_path:
            with open(save_path, 'w', encoding='utf8') as f:
                f.write('')

        results = dict()
        for evaluator in self._evaluators:
            results.update(evaluator.evaluate(save_path))

        return results

    @staticmethod
    def load_task_build_config_and_noise_pool(
        dir_path: str,
        task_name: str,
    ) -> tuple[BuildConfig, NoisePool]:
        build_config = BuildConfig()

        test_dir = os.path.join(dir_path, 'test')
        constellation = json_load(os.path.join(test_dir, 'constellation.json'))
        build_config.update(constellation)

        orbits = json_load(os.path.join(test_dir, 'orbits.json'))
        build_config.update(orbits)

        tasks = json_load(os.path.join(test_dir, f'{task_name}_tasks.json'))
        build_config.update(tasks)

        noise_pool = torch.load(
            os.path.join(test_dir, f'{task_name}_noise_pool.pth'),
            map_location=torch.get_default_device(),
        )
        return build_config, noise_pool

    @staticmethod
    def get_aie_evaluators() -> list[BaseEvaluator]:
        evaluators_config = [
            dict(type='PowerEvaluator'),
            dict(type='AIE'),
        ]
        evaluators = [EvaluatorRegistry.build(c) for c in evaluators_config]
        return evaluators

    @staticmethod
    def get_time_domain_evaluators(
            threshold: float = 0.01) -> list[BaseEvaluator]:
        evaluators_config = [
            dict(type='TDPA', threshold=threshold),
            dict(type='PowerEvaluator'),
        ]
        evaluators = [EvaluatorRegistry.build(c) for c in evaluators_config]
        return evaluators


@BenchmarkRegistry.register_()
class AttitudeMaintainBenchmark(Benchmark):

    def __init__(
        self,
        dir_path: str,
        **kwargs,
    ) -> None:
        evaluators = self.get_aie_evaluators()
        build_config, noise_pool = self.load_task_build_config_and_noise_pool(
            dir_path,
            'attitude_maintain',
        )
        super().__init__(
            evaluators,
            AttitudeMaintainEnvironment,
            build_config,
            with_window=True,
            noise_pool=noise_pool,
            **kwargs,
        )


@BenchmarkRegistry.register_()
class LocationPointingBenchmark(Benchmark):

    def __init__(
        self,
        dir_path: str,
        initial_ephemeris_path: str,
        threshold: float = 0.01,
        *args,
        **kwargs,
    ) -> None:
        build_config, noise_pool = self.load_task_build_config_and_noise_pool(
            dir_path,
            'location_pointing',
        )
        super().__init__(
            self.get_time_domain_evaluators(threshold),
            LocationPointingEnvironment,
            build_config,
            initial_ephemeris_path=initial_ephemeris_path,
            noise_pool=noise_pool,
            *args,
            **kwargs,
        )


@BenchmarkRegistry.register_()
class SpacecraftPointingBenchmark(Benchmark):

    def __init__(
        self,
        dir_path: str,
        threshold: float = 0.01,
        **kwargs,
    ) -> None:
        build_config, noise_pool = self.load_task_build_config_and_noise_pool(
            dir_path,
            'spacecraft_pointing',
        )
        super().__init__(
            self.get_time_domain_evaluators(threshold),
            SpacecraftPointingEnvironment,
            build_config,
            noise_pool=noise_pool,
            **kwargs,
        )
