import argparse
import pathlib
import sys
from typing import cast

import torch
from attitude_maneuver.callbacks import ComposedCallback
from attitude_maneuver.environment import AttitudeControlEnvironment
from attitude_maneuver.model import MLPPIDConfigure
from attitude_maneuver.registries import CallbackRegistry
from attitude_maneuver.runner import ControllerRunner
from todd.bases.configs import Config
from todd.configs import PyConfig
from todd.patches.py_ import DictAction
from todd.utils import init_seed


def log(
    runner: ControllerRunner,
    args: argparse.Namespace,
    config: PyConfig,
) -> None:
    runner.logger.info("Command\n" + ' '.join(sys.argv))
    runner.logger.info(f"Args\n{vars(args)}")
    runner.logger.info(f"Config\n{config.dumps()}")

    if 'config' in args:
        config_name = cast(pathlib.Path, args.config).name
        config.dump(runner.work_dir / config_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=pathlib.Path,
        help='Path to the config file.',
    )
    parser.add_argument(
        '--load',
        '-l',
        type=str,
        default=None,
        help="Use a specified initial model state dict",
    )
    parser.add_argument(
        '--work-dir',
        type=pathlib.Path,
        default=pathlib.Path('./work_dir/test'),
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    init_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    config = PyConfig.load(args.config)
    config.override(args.override)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

    # build model
    model = MLPPIDConfigure(**config.runner.model)
    if args.load:
        model_state_dict = torch.load(args.load)
        model.load_state_dict(model_state_dict)

    # build optimizer
    optim_config = dict()
    optim_config.update(config.runner.optimizer)

    optim_fn = eval(optim_config.pop('type', 'torch.optim.SGD'))
    optimizer = optim_fn(model.parameters(), **optim_config)

    env = AttitudeControlEnvironment(**config.runner.environment)

    callbacks = CallbackRegistry.build(
        Config(
            type=ComposedCallback.__name__, callbacks=config.runner.callbacks
        )
    )
    config.runner.work_dir = cast(pathlib.Path, args.work_dir)

    trainer = ControllerRunner(
        model,
        callbacks,
        config.runner,
        env,
        optimizer,
    )
    log(trainer, args, config)

    if args.debug:
        torch.set_anomaly_enabled(True)

    trainer.train()
    # torch.set_anomaly_enabled(False)
