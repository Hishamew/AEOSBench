import argparse

import torch

from satsim.architecture import Timer, constants
from satsim.simulation.gravity import Ephemeris, SpiceInterface


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
    timer = Timer()
    spice_interface = SpiceInterface(
        timer=timer,
        utc_time_init=constants.UTC_TIME_START,
    )
    timer.reset()
    _, (ephemeris, ) = spice_interface('EARTH')
    torch.save(ephemeris, args.output_file)


if __name__ == '__main__':
    main()
