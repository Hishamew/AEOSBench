import argparse
import os
import pathlib
import random
from copy import deepcopy

import todd
import torch
import torch.nn as nn
from attitude_maneuver.model import MLPPIDConfigure
from satsim.architecture import constants
from todd.configs import PyConfig

from constellation import (
    SATELLITES_ROOT,
    TASKSETS_ROOT,
    Controller,
    TaskManager,
)
from constellation.algorithms import OptimalAlgorithm
from constellation.callbacks import ComposedCallback
from constellation.data import (
    Constellation,
    MRPControl,
    Satellite,
    Satellites,
    TaskSet,
)
from constellation.environments import SatsimEnvironment
from constellation.evaluators import CompletionRateEvaluator

RANK = int(os.environ['RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])

TASKSET_PATH = TASKSETS_ROOT / 'mrp.json'
TASKSET = TaskSet.load(str(TASKSET_PATH))


def reconfigure_pid(
    constellation: Constellation,
    configurer: MLPPIDConfigure,
) -> Constellation:
    sc_masses = []
    sc_inertias = []
    rw_inertias = []

    for satellite in constellation.sort():
        sc_masses.append(satellite.mass)
        sc_inertia = satellite.inertia
        sc_inertia = [sc_inertia[0], sc_inertia[4], sc_inertia[8]]
        sc_inertias.append(torch.tensor(sc_inertia))
        rw_inertia = [
            rw.max_momentum / (6000 * constants.RPM)
            for rw in satellite.reaction_wheels
        ]
        rw_inertias.append(torch.tensor(rw_inertia))
    sc_masses = torch.tensor(sc_masses)
    sc_inertia = torch.stack(sc_inertias)
    rw_inertia = torch.stack(rw_inertias)
    pid_gains: torch.Tensor = configurer(sc_masses, sc_inertia, rw_inertia)
    k, ki, p, integral_limit = pid_gains.unbind(dim=-1)
    satellites = [
        Satellite(
            sat.id_,
            sat.inertia,
            sat.mass,
            sat.center_of_mass,
            sat.orbit_id,
            sat.orbit,
            sat.solar_panel,
            sat.sensor,
            sat.battery,
            sat.reaction_wheels,
            MRPControl(
                k=k[i].item(),
                ki=ki[i].item(),
                p=p[i].item(),
                integral_limit=integral_limit[i].item(),
            ),
            sat.true_anomaly,
            sat.mrp_attitude_bn,
        ) for i, sat in enumerate(constellation.sort())
    ]
    new_constellation = Constellation({
        satellite.id_: satellite
        for satellite in satellites
    })
    return new_constellation


def generate_satellites(
    split: str,
    n: int,
    completion_rate_threshold: float,
    configurer: MLPPIDConfigure,
) -> None:
    satellites_root: pathlib.Path = SATELLITES_ROOT / split
    max_id = max(
        (
            int(satellite_path.stem)
            for satellite_path in satellites_root.iterdir()
        ),
        default=-1,
    )

    for i in range(RANK, n, WORLD_SIZE):
        if i <= max_id:
            continue

        constellation = Constellation.sample_mrp()
        constellation = reconfigure_pid(constellation, configurer)
        environment = SatsimEnvironment(
            constellation=constellation, all_tasks=TASKSET, backend='cpu'
        )
        task_manager = TaskManager(timer=environment.timer, taskset=TASKSET)
        callbacks = ComposedCallback(
            callbacks=[
                CompletionRateEvaluator(),
                # StateCollector(
                #     work_dir=pathlib.Path('log'),
                #     namespace=dict(
                #         attitude_error='_location_pointing.attitude_BR_old',
                #         integral_sigma='_mrp_control.integral_sigma',
                #     ),
                #     default=dict(
                #         attitude_error=torch.zeros(
                #             1, 3, device=environment._backend
                #         ),
                #     )
                # ),
                # MemoCollector(
                #     work_dir=pathlib.Path('log'),
                #     namespace=dict(
                #         assignment=torch.tensor,
                #         is_visible=torch.stack,
                #         max_progress=torch.stack,
                #     )
                # )
            ],
        )
        controller = Controller(
            f'{i:05}',
            environment=environment,
            task_manager=task_manager,
            callbacks=callbacks,
        )

        algorithm = OptimalAlgorithm(timer=environment.timer)
        algorithm.prepare(environment, task_manager)

        # try:

        controller.run(algorithm, progress_bar=False, max_time_step=7200)
        # except Exception as e:
        #     todd.logger.error("rank %d failed %d: %s", RANK, i, e)
        #     continue

        completion_rate = controller.memo['metrics']['CR']
        todd.logger.info("rank %d finished %d with %s", RANK, i, completion_rate)  # noqa: E501 yapf: disable
        if completion_rate > completion_rate_threshold:
            constellation.dump(str(satellites_root / f'{i}.json'))
        else:
            constellation.dump(f'log/failed_{i}.json')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.99)
    parser.add_argument('--config', type=pathlib.Path, default=None)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--progress-bar', action='store_true')

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    model_config = PyConfig.load(args.config)
    model_config = model_config.runner.model

    configurer = MLPPIDConfigure(**model_config)
    # device = torch.device(RANK % torch.cuda.device_count())
    # torch.cuda.set_device(device)
    configurer.load_state_dict(
        torch.load(args.ckpt, map_location='cpu'),
    )

    generate_satellites(
        'train',
        10_000,
        args.threshold,
        configurer,
    )
    generate_satellites(
        'val_unseen',
        2_000,
        args.threshold,
        configurer,
    )
    generate_satellites(
        'test',
        2_000,
        args.threshold,
        configurer,
    )


if __name__ == "__main__":
    main()
