import argparse
import os
import random

import torch
from todd.configs import PyConfig
from todd.loggers import master_logger
from todd.patches.py_ import DictAction

from .algorithm import ModelRegistry
from .algorithm.base import BaseModel
from .algorithm.benchmark import Benchmark
from .algorithm.registry import BenchmarkRegistry


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
        type=int,
        default=42,
    )
    parser.add_argument(
        '--save_path',
        '-s',
        type=str,
        default='test/benchmark/test.txt',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.config:
        config = PyConfig.load(args.config)
    else:
        config = PyConfig.loads(args.from_script)
    if args.override:
        config.override(args.override)

    model: BaseModel = ModelRegistry.build(config.model)
    if args.ckpt:
        state_dict = torch.load(args.ckpt)
        model.load_state_dict(state_dict)
        master_logger.info(f"Checkpoint Loaded: {args.ckpt}")

    benchmark: Benchmark = BenchmarkRegistry.build(config.benchmark)
    dir_path = os.path.dirname(args.save_path)
    os.makedirs(dir_path, exist_ok=True)

    benchmark.evaluate(model, args.save_path)
