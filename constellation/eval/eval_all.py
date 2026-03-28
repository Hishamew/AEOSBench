import argparse
import importlib
import logging
import os
import pathlib
from functools import partial
from itertools import count

import todd
import torch
from todd.configs import PyConfig
from todd.loggers import Formatter
from todd.patches.py_ import DictAction, json_dump
from todd.patches.torch import load_state_dict, load_state_dict_
from todd.utils import init_seed

from .controller_wrapper import ControllerWrapper
from .dummy import DummyVecControllerEnv
from .model_algorithm import ModelAlgorithm


def build(
    num_env_per_rank: int,
    *args,
    **kwargs,
) -> DummyVecControllerEnv:
    assert num_env_per_rank > 0, "world_size must be greater than 0"
    return DummyVecControllerEnv([
        partial(
            ControllerWrapper,
            *args,
            world_size=int(os.environ['WORLD_SIZE']) * num_env_per_rank,
            rank=int(os.environ['RANK']) * num_env_per_rank + i,
            **kwargs,
        ) for i in range(num_env_per_rank)
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('name')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--config-options', action=DictAction, default=dict())
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--load-model-from', nargs='+', default=[])
    parser.add_argument('--retry-from', type=pathlib.Path, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = PyConfig.load(args.config, **args.config_options)
    config.override(args.override)
    init_seed(args.seed)

    for custom_import in config.get('custom_imports', []):
        importlib.import_module(custom_import)

    work_dir = pathlib.Path('work_dirs') / f'rl_eval_{args.name}'
    work_dir.mkdir(parents=True, exist_ok=True)

    gen_trajectory_dir = work_dir / config.environment.split
    gen_trajectory_dir.mkdir(parents=True, exist_ok=True)

    environment = build(
        retry_from=args.retry_from,
        gen_trajectory_dir=gen_trajectory_dir,
        **config.environment,
    )

    device = torch.device(int(os.environ['RANK']) % torch.cuda.device_count())
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    policy = ModelAlgorithm(greedy=True)
    if args.load_model_from != []:
        load_state_dict(
                policy.actor,  # type: ignore[arg-type]
                load_state_dict_(args.load_model_from),  # type: ignore[arg-type]
                strict=False,
            )

    todd.logger.handlers.clear()
    if os.environ['RANK'] == '0':
        handler = logging.StreamHandler()
        handler.setFormatter(Formatter())
        todd.logger.addHandler(handler)

    log_dir = work_dir / 'log'
    log_dir.mkdir(parents=False, exist_ok=True)

    log_path = log_dir / f"{os.environ['RANK']}.log"

    handler = logging.FileHandler(log_path)
    handler.setFormatter(Formatter())
    todd.logger.addHandler(handler)

    environment.reset()
    for i in count():
        if environment.all_done:
            todd.logger.info(
                "rank %s step %d all done",
                os.environ['RANK'],
                i,
            )
            break
        if i % config.log_interval == 0:
            for all_done, rank in zip(
                environment.get_attr('all_done'),
                environment.get_attr('rank'),
            ):
                todd.logger.info(
                    f"Env id {rank} step {i}: {'done' if all_done else 'running'}"
                )

        observations = environment.get_observations()
        actions = policy.step(observations)
        environment.step(actions)
