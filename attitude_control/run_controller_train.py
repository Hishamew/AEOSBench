import argparse
import random

import torch
from todd.configs import PyConfig
from todd.loggers import master_logger
from todd.patches.py_ import DictAction
from todd.registries.patches import LRSchedulerRegistry
from torch import nn

from .algorithm import ControllerRunner, ModelRegistry
from .algorithm.benchmark import Benchmark
from .algorithm.environment.base_environment import AttitudeControlEnvironment
from .algorithm.registry import BenchmarkRegistry, EnvironmentRegistry


class OptimWrapper:

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        self._optim = optimizer
        self._scheduler = scheduler

    def step(self) -> None:
        self._optim.step()
        self._scheduler.step()
        master_logger.info(
            f'Current lr: {self._scheduler.get_last_lr()[0]:.6f}')

    def zero_grad(self) -> None:
        self._optim.zero_grad()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=str,
        help='Path to the config file.',
    )
    parser.add_argument(
        '--load',
        '-l',
        type=str,
        default=None,
        help="Use a specified initial model state dict",
    )
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = PyConfig.load(args.config)
    config.override(args.override)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    # build model
    model: nn.Module = ModelRegistry.build(config.model)
    if args.load:
        model_state_dict = torch.load(args.load)
        model.load_state_dict(model_state_dict)

    # build optimizer
    optim_config = dict()
    optim_config.update(config.optim)
    scheduler_config = dict()
    scheduler_config.update(optim_config.pop('scheduler', dict()))

    optim_fn = eval(optim_config.pop('type', 'torch.optim.SGD'))
    optimizer = optim_fn(model.parameters(), **optim_config)

    if scheduler_config:
        # scheduler_config
        # scheduler_fn = eval(scheduler_config.pop('type'))
        # scheduler = scheduler_fn(optimizer, **scheduler_config)
        scheduler = LRSchedulerRegistry.build(scheduler_config,
                                              optimizer=optimizer)
        optimizer = OptimWrapper(optimizer, scheduler)

    benchmark: Benchmark = BenchmarkRegistry.build(config.benchmark)
    env: AttitudeControlEnvironment = EnvironmentRegistry.build(
        config.environment)

    trainer = ControllerRunner(
        model,
        config,
        benchmark,
        env,
        optimizer,
    )
    # torch.set_anomaly_enabled(True)
    trainer.train()
    # torch.set_anomaly_enabled(False)
