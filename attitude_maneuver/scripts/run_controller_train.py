import argparse
from pprint import pformat

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

    trainer = ControllerRunner(
        model,
        callbacks,
        config.runner,
        env,
        optimizer,
    )
    trainer.logger.info(pformat(config))

    config.dump(trainer.work_dir / 'config.py')
    if args.debug:
        torch.set_anomaly_enabled(True)

    trainer.train()
    # torch.set_anomaly_enabled(False)
