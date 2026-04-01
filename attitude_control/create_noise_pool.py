import argparse
import os.path as osp

import torch
from attitude_control.algorithm.constellation.pertubation import (
    NoisePool,
    PerturbationForceTorque,
    SensorErrorModel,
)
from todd.loggers import master_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Create Noise Pool')
    parser.add_argument(
        'dir_path',
        type=str,
        help='Output filename for the noise pool',
    )
    parser.add_argument(
        '--length',
        type=int,
        default=720,
        help='Sampling length',
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        default=500,
        help='Number of environments',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0x1badcad1,
        help='Random seed for noise generation',
    )
    return parser.parse_args()


def generate_noise_pool(
    num_envs: int,
    length: int,
    generator: torch.Generator,
) -> NoisePool:
    attitude_error = SensorErrorModel.sample_sensor_error(
        num_envs,
        length,
        generator,
    )
    transition_error = SensorErrorModel.sample_sensor_error(
        num_envs,
        length,
        generator,
    )

    force, torque = PerturbationForceTorque.sample_perturbation(
        num_envs,
        length,
        generator,
    )

    noise_pool = NoisePool(
        transition_error=transition_error,
        attitude_error=attitude_error,
        perturbation_force=force,
        perturbation_torque=torque,
    )
    return noise_pool


def main():
    args = parse_args()

    master_logger.info(
        f"Generating noise pool: {args.num_envs} envs, length {args.length}..."
    )
    gen = torch.Generator(device=torch.get_default_device())
    gen.manual_seed(args.seed)

    attitude_maintain_noise_pool = generate_noise_pool(
        num_envs=args.num_envs,
        length=args.length,
        generator=gen,
    )
    location_pointing_noise_pool = generate_noise_pool(
        num_envs=args.num_envs,
        length=args.length,
        generator=gen,
    )
    spacecraft_pointing_noise_pool = generate_noise_pool(
        num_envs=args.num_envs,
        length=args.length,
        generator=gen,
    )
    torch.save(
        attitude_maintain_noise_pool,
        osp.join(args.dir_path, 'attitude_maintain_noise_pool.pth'),
    )
    torch.save(
        location_pointing_noise_pool,
        osp.join(args.dir_path, 'location_pointing_noise_pool.pth'),
    )
    torch.save(
        spacecraft_pointing_noise_pool,
        osp.join(args.dir_path, 'spacecraft_pointing_noise_pool.pth'),
    )

    master_logger.info(f"Successfully saved noise pool to {args.dir_path}")


if __name__ == '__main__':
    main()
