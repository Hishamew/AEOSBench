import argparse

import torch
from todd.configs import PyConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to save the extracted initial ephemeris.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = PyConfig.load('algo/configs/default.py')
    config.dump(args.output_file)


if __name__ == '__main__':
    main()
