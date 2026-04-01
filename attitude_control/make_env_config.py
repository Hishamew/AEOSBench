import argparse

from todd.configs import PyConfig
from todd.patches.py_ import json_dump

from algo.algorithm.environment import AttitudeControlEnvironment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'initial_ephemeris',
        type=str,
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to save the extracted initial ephemeris.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = AttitudeControlEnvironment(
        1,
        args.initial_ephemeris,
    )
    env.reset()
    build_config, _ = env.state_dict()

    json_dump(build_config, args.output_file, indent=4)


if __name__ == '__main__':
    main()
