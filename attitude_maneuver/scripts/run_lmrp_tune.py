import argparse
import pathlib
import sys
from typing import cast

import torch
from attitude_maneuver.callbacks import ComposedCallback
from attitude_maneuver.environment import AttitudeControlEnvironment
from attitude_maneuver.model import LearnablePIDConfigure
from attitude_maneuver.registries import CallbackRegistry
from attitude_maneuver.runner import ControllerRunner
from todd.bases.configs import Config
from todd.configs import PyConfig
from todd.patches.py_ import DictAction
from todd.utils import init_seed

from constellation.data.constellations import Constellation


def log(
    runner: ControllerRunner,
    args: argparse.Namespace,
    config: PyConfig,
) -> None:

    runner.logger.info("Command\n" + ' '.join(sys.argv))
    runner.logger.info(f"Args\n{vars(args)}")
    runner.logger.info(f"Config\n{config.dumps()}")
    config.runner.work_dir = str(config.runner.work_dir)

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
        '--work-dir',
        type=pathlib.Path,
        default=pathlib.Path('./work_dir/test'),
    )
    parser.add_argument('--constellation', type=pathlib.Path, required=True)
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        '--save-path',
        type=pathlib.Path,
        required=True,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    init_seed(args.seed)

    # device = 0
    # torch.cuda.set_device(device)
    # torch.set_default_device(device)
    torch.set_default_device('cpu')

    config = PyConfig.load(args.config)
    config.override(args.override)

    constellation = Constellation.load(str(args.constellation))
    num_sats = len(constellation)

    # build model
    model = LearnablePIDConfigure(**config.runner.model, num_sats=num_sats)

    # build optimizer
    optim_config = dict()
    optim_config.update(config.runner.optimizer)

    optim_fn = eval(optim_config.pop('type'))
    optimizer = optim_fn(model.parameters(), **optim_config)

    env = AttitudeControlEnvironment(**config.runner.environment)
    env.constellation = constellation

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

    trainer.run()
    final_constellation: Constellation = trainer.memo['constellation']
    final_constellation.dump(str(args.save_path))
