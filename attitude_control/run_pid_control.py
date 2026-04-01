import argparse
import random

import torch
from todd.configs import PyConfig
from todd.patches.py_ import DictAction

from satsim.architecture.timer import Timer

from .algorithm import ControllerRunner, ModelRegistry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=str,
        help='Path to the config file.',
    )
    parser.add_argument(
        'pid_arguments',
        type=float,
        nargs=3,
        help='PID controller arguments: dt, k, ki, p.',
    )
    parser.add_argument(
        '--dt',
        type=float,
        help='Time step for the timer.',
        default=1.0,
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

    k, ki, p = args.pid_arguments

    pid_config = dict(
        type='MRPFeedbackController',
        dt=args.dt,
        k=k,
        ki=ki,
        p=p,
    )

    timer = Timer(1.)
    model = ModelRegistry.build(pid_config)

    trainer = ControllerRunner(model, config)

    trainer.test(args.tag)
    model.reset()
    trainer.benchmark()
