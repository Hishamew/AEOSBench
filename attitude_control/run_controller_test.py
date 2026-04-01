import argparse
import random

import torch
from todd.configs import PyConfig
from todd.loggers import master_logger
from todd.patches.py_ import DictAction
from torch import nn

from .algorithm import ControllerRunner, ModelRegistry
from .algorithm.benchmark import Benchmark
from .algorithm.environment.base_environment import AttitudeControlEnvironment
from .algorithm.registry import BenchmarkRegistry, EnvironmentRegistry


def parse_args():
    parser = argparse.ArgumentParser()
    mutex_group = parser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument(
        '--config',
        '-c',
        type=str,
        help='Path to the config file.',
    )
    mutex_group.add_argument(
        '--from_script',
        '-f',
        type=str,
        help='Form config from script',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--override',
        action=DictAction,
        default=dict(),
    )
    parser.add_argument(
        '--seed',
        '-s',
        type=int,
        default=42,
    )
    parser.add_argument('--tag', '-t', type=str, default='test')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = PyConfig.load(args.config)
    config.override(args.override)

    model: nn.Module = ModelRegistry.build(config.model)
    if args.ckpt:
        state_dict = torch.load(args.ckpt)
        model.load_state_dict(state_dict)
        master_logger.info(f"Checkpoint Loaded: {args.ckpt}")

    benchmark: Benchmark = BenchmarkRegistry.build(config.benchmark)
    env: AttitudeControlEnvironment = EnvironmentRegistry.build(
        config.environment)
    trainer = ControllerRunner(
        model,
        config,
        benchmark,
        env,
    )
    trainer.test(args.tag)
    trainer.benchmark(args.tag)
