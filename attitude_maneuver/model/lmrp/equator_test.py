__all__ = [
    'EquatorTestValidator',
]
import os
import pathlib
from multiprocessing import Pool, cpu_count

import spiceypy
import todd
import torch
from todd.patches.py_ import json_dump

from constellation import TASKSETS_ROOT, Controller, TaskManager
from constellation.algorithms import OptimalAlgorithm
from constellation.callbacks import ComposedCallback
from constellation.data import Constellation, MRPControl, Satellite, TaskSet
from constellation.environments import SatsimEnvironment
from constellation.evaluators import CompletionRateEvaluator

from ...callbacks import BaseCallback
from ...registries import CallbackRegistry
from .lmrp import LearnablePIDConfigure

TASKSET_PATH = TASKSETS_ROOT / 'mrp.json'
TASKSET = TaskSet.load(str(TASKSET_PATH))


def init_worker():
    spiceypy.kclear()
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # NOTE: This manual kernel furnish is required to avoid spiceypy kernel pool conflict.
    dir_path = 'third_party/Sat-Sim-pytorch/satsim/simulation/gravity/spice_kernel'
    file_names = [
        "de430.bsp",
        "naif0012.tls",
        "de-403-masses.tpc",
        "pck00010.tpc",
    ]
    for file_name in file_names:
        spiceypy.furnsh(os.path.join(dir_path, file_name))


def recover_to_equator_task(
    constellation: Constellation,
    ks: list[float],
    kis: list[float],
    ps: list[float],
    integral_limits: list[float],
) -> list[Constellation]:
    equator_constellations = []
    for sat, k, ki, p, integral_limit in zip(
        constellation.sort(),
        ks,
        kis,
        ps,
        integral_limits,
    ):
        sat_mrp = Constellation.sample_mrp()[0]
        mrp_control = MRPControl(
            k,
            ki,
            p,
            integral_limit,
        )
        recovered_sat = Satellite(
            sat_mrp.id_,
            sat.inertia,
            sat.mass,
            (0.0, 0.0, 0.0),
            sat_mrp.orbit.id_,
            sat_mrp.orbit,
            sat_mrp.solar_panel,
            sat_mrp.sensor,
            sat_mrp.battery,
            sat.reaction_wheels,
            mrp_control,
            0.0,
            (0.0, 0.0, 0.0),
        )
        equator_constellations.append(
            Constellation({recovered_sat.id_: recovered_sat})
        )
    return equator_constellations


def evaluate_satellites(
    constellation: Constellation,
    id_: int,
    skip: bool,
) -> float:
    if skip:
        todd.logger.info(f"{id_} is done, skipping evaluation.")
        return 1.0
    environment = SatsimEnvironment(
        constellation=constellation,
        all_tasks=TASKSET,
        backend='cpu',
        skip_kernel_furn=True,
    )
    task_manager = TaskManager(timer=environment.timer, taskset=TASKSET)
    callbacks = ComposedCallback(
        callbacks=[CompletionRateEvaluator()],
    )
    controller = Controller(
        f'{id_:05}',
        environment=environment,
        task_manager=task_manager,
        callbacks=callbacks,
    )

    algorithm = OptimalAlgorithm(timer=environment.timer)
    algorithm.prepare(environment, task_manager)

    controller.run(algorithm, progress_bar=id_ == 0, max_time_step=7200)
    completion_rate = controller.memo['metrics']['CR']
    todd.logger.info(f"finished {id_} with {completion_rate}")
    return completion_rate


@CallbackRegistry.register_()
class EquatorTestValidator(BaseCallback):

    def __init__(
        self,
        *args,
        threshold: float = 0.99,
        interval: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._threshold = threshold
        self._interval = interval

    @property
    def should_run_test(self) -> bool:
        return self.runner.episode % self._interval == 0

    @property
    def model(self) -> LearnablePIDConfigure:
        return self.runner.model

    @property
    def work_dir(self) -> pathlib.Path:
        return self.runner.work_dir / 'test_result'

    def bind(self, *args, **kwargs):
        super().bind(*args, **kwargs)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _run_test(self) -> None:
        constellation = self.runner.environment.constellation
        with torch.no_grad():
            params = self.runner.model()
        ks, kis, ps, integral_limits = map(
            lambda x: x.cpu().tolist(), params.unbind(-1)
        )
        recovered_constellation = recover_to_equator_task(
            constellation,
            ks,
            kis,
            ps,
            integral_limits,
        )

        total_sat = len(recovered_constellation)

        self.runner.logger.info(
            'Running evaluation for %d satellites', total_sat
        )
        max_processes = min(total_sat, cpu_count())
        with Pool(processes=max_processes, initializer=init_worker) as pool:
            tasks = [
                (recovered_constellation[i], i, self.model.done[i].item())
                for i in range(total_sat)
            ]
            crs = pool.starmap(evaluate_satellites, tasks)

        before_done = self.model.done
        new_done = before_done.new_tensor([cr > self._threshold for cr in crs])
        done = new_done.bitwise_or(before_done)
        self.model.done = done

        not_done_list = (~done).nonzero().squeeze().tolist()

        self.runner.logger.info(
            "Episode %d: %d/%d satellites done | Not done idx: %s",
            self.runner.episode,
            self.model.done.sum().item(),
            len(self.model.done),
            not_done_list,
        )

        json_file = self.work_dir / f'{self.runner.tag}.json'
        json_dump(
            dict(done=done.tolist(), crs=crs),
            str(json_file),
            indent=4,
        )

    def after_episode(self):
        if not self.should_run_test:
            return

        self._run_test()

    def after_run(self):
        if self.model.done.all():
            self.runner.logger.info(
                "All satellites are done, skipping final test."
            )
            return
        self._run_test()
