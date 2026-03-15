import argparse
import os
import pathlib

import todd
import torch
import torch.nn as nn

from constellation import (
    SATELLITES_ROOT,
    TASKSETS_ROOT,
    Controller,
    TaskManager,
)
from constellation.algorithms import OptimalAlgorithm
from constellation.callbacks import ComposedCallback
from constellation.data import Constellation, MRPControl, Satellite, TaskSet
from constellation.environments import SatsimEnvironment
from constellation.evaluators import (
    CompletionRateEvaluator,
    PerCompletionRateEvaluator,
)

RANK = int(os.environ['RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])

TASKSET_PATH = TASKSETS_ROOT / 'mrp.json'
TASKSET = TaskSet.load(str(TASKSET_PATH))


class MLPPIDConfigure(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        task_invariant: bool = True,
    ) -> None:
        super().__init__()
        self._input_dim = 7 if task_invariant else 12
        self._hidden_dim = hidden_dim
        self._task_invariant = task_invariant

        self.input_projection = nn.Linear(
            self._input_dim,
            hidden_dim,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, 3),
        )
        self._softplus = nn.Softplus()  # linear[-1] -> [0, +oo]

    def forward(
        self,
        sc_mass: torch.Tensor,
        sc_inertia: torch.Tensor,
        rw_inertia: torch.Tensor,
    ) -> torch.Tensor:
        feature = torch.cat(
            [
                rw_inertia,
                sc_mass.unsqueeze(-1),
                sc_inertia,
            ],
            dim=-1,
        )
        x = self.input_projection(feature)
        raw_gains = self.mlp(x)

        pid_gains: torch.Tensor = self._softplus(raw_gains)
        return pid_gains


def reconfigure_pid(
    constellation: Constellation,
    configurer: MLPPIDConfigure,
) -> Constellation:
    sc_masses = []
    sc_inertias = []

    # fixed rw inertia
    rw_inertia = 12 / (6000 * 2 * torch.pi / 60)
    rw_inertias = [rw_inertia, rw_inertia, rw_inertia]
    for satellite in constellation.sort():
        sc_masses.append(satellite.mass)
        sc_inertia = satellite.inertia
        sc_inertia = [sc_inertia[0], sc_inertia[4], sc_inertia[8]]
        sc_inertias.append(torch.tensor(sc_inertia))

    sc_masses = torch.tensor(sc_masses)
    sc_inertia = torch.stack(sc_inertias)
    rw_inertia = torch.tensor(rw_inertias
                              ).unsqueeze(0).expand(len(constellation), -1)

    pid_gains: torch.Tensor = configurer(sc_masses, sc_inertia, rw_inertia)
    k, ki, p = pid_gains.unbind(dim=-1)
    ki = ki * 1e-4
    breakpoint()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.99)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--num_satellites', type=int, default=512)
    args = parser.parse_args()
    return args


def generate_satellites(
    split: str,
    n: int,
    completion_rate_threshold: float,
    configurer: MLPPIDConfigure,
    num_satellites: int,
) -> None:
    satellites_root: pathlib.Path = SATELLITES_ROOT / split
    max_id = max(
        (
            int(satellite_path.stem)
            for satellite_path in satellites_root.iterdir()
        ),
        default=-1,
    )

    fake_batch = []
    for i in range(RANK, n, WORLD_SIZE):
        if i <= max_id:
            continue

        fake_batch.append(i)
        if len(fake_batch) < num_satellites:
            continue

        constellation = Constellation.sample_mrp(len(fake_batch))
        constellation = reconfigure_pid(constellation, configurer)
        environment = SatsimEnvironment(
            constellation=constellation,
            all_tasks=TASKSET,
        )
        task_manager = TaskManager(timer=environment.timer, taskset=TASKSET)
        callbacks = ComposedCallback(
            callbacks=[
                PerCompletionRateEvaluator(),
            ],
        )
        controller = Controller(
            pathlib.Path(__file__).stem,
            environment=environment,
            task_manager=task_manager,
            callbacks=callbacks,
        )

        algorithm = OptimalAlgorithm(timer=environment.timer)
        algorithm.prepare(environment, task_manager)

        try:
            controller.run(algorithm, progress_bar=False, max_time_step=7200)
        except Exception as e:
            todd.logger.error("rank %d failed %d: %s", RANK, i, e)
            continue

        completion_rate: list[float] = controller.memo['metrics']['CR_persat']
        for j, cr in zip(fake_batch, completion_rate):
            todd.logger.info("rank %d finished %d with %s", RANK, j, cr)  # noqa: E501 yapf: disable
            if cr > completion_rate_threshold:
                constellation.dump(str(satellites_root / f'{j}.json'))


def main() -> None:
    args = parse_args()
    configurer = MLPPIDConfigure(hidden_dim=128, task_invariant=True)
    configurer.load_state_dict(torch.load(args.ckpt))

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
