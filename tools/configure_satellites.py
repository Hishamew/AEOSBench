import argparse
import os
import pathlib
import random
from copy import deepcopy

import todd
import torch
import torch.nn as nn
from satsim.architecture import constants

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


class InputNormalizer(nn.Module):

    def __init__(
        self,
        shape: int | list[int],
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        shape = [shape] if isinstance(shape, int) else shape
        self.register_buffer('_running_mean', torch.zeros(*shape))
        self.register_buffer('_running_var', torch.ones(*shape))
        self.register_buffer('_count', torch.tensor(epsilon))
        self._epsilon = epsilon

    @property
    def running_mean(self) -> torch.Tensor:
        return self.get_buffer('_running_mean')

    @property
    def running_var(self) -> torch.Tensor:
        return self.get_buffer('_running_var')

    @property
    def count(self) -> torch.Tensor:
        return self.get_buffer('_count')

    def forward(self, batched_input: torch.Tensor):

        result = (batched_input - self.running_mean
                  ) / torch.sqrt(self.running_var + self._epsilon)

        return result


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

        self.input_projection = nn.Linear(self._input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, 3),
        )

        self._input_normalizer = InputNormalizer(self._input_dim)
        self._runtime_normalizer = deepcopy(self._input_normalizer)

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
        feature = self._runtime_normalizer(feature)

        x = self.input_projection(feature)
        raw_gains = self.mlp(x)

        pid_gains: torch.Tensor = torch.exp(raw_gains)
        return pid_gains


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
    k, ki, p = pid_gains.unbind(dim=-1)
    ki = ki * 1e-4
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
                integral_limit=random.uniform(0.05, 0.2),
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
            constellation=constellation,
            all_tasks=TASKSET,
        )
        task_manager = TaskManager(timer=environment.timer, taskset=TASKSET)
        callbacks = ComposedCallback(
            callbacks=[
                CompletionRateEvaluator(),
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

        # try:
        controller.run(algorithm, progress_bar=False, max_time_step=7200)
        # except Exception as e:
        #     todd.logger.error("rank %d failed %d: %s", RANK, i, e)
        #     continue

        completion_rate = controller.memo['metrics']['CR']
        todd.logger.info("rank %d finished %d with %s", RANK, i, completion_rate)  # noqa: E501 yapf: disable
        if completion_rate > completion_rate_threshold:
            constellation.dump(str(satellites_root / f'{i}.json'))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.99)
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    configurer = MLPPIDConfigure(hidden_dim=256, task_invariant=True)
    device = torch.device(RANK % torch.cuda.device_count())
    torch.cuda.set_device(device)
    configurer.load_state_dict(
        torch.load(args.ckpt, map_location=device),
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
