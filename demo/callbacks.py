import pathlib
from collections import defaultdict

import torch
from satsim.architecture import constants
from satsim.utils import PCPF2LLA
from satsim.utils import dcm_to_eulerparameters as dcm2q
from satsim.utils import mrp_to_rotation_matrix as mrp2dcm
from todd.patches.py_ import json_dump

from constellation.callbacks import BaseCallback

from .interface import SatelliteData, Socket, TaskData


class RecordCallback(BaseCallback):

    def __init__(self, *args, log_dir: pathlib.Path, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_dir = log_dir

    @property
    def progress(self) -> torch.Tensor:
        # NOTE: TaskManager will not track the progress
        # of succeeded task and invalid task, to record all task progress
        # we sustain a progress property here
        return self._progress

    @property
    def completion(self) -> list[float]:
        duration = [
            task.duration for task in self.controller.task_manager.taskset
        ]
        return [p / d * 100 for p, d in zip(self.progress, duration)]

    def after_step(self):
        time_stamp = self.controller.environment.timer.time
        task_managers = self.controller.task_manager

        constellation = self.controller.environment.get_constellation()
        earth_rotation_pn = self.controller.environment.get_earth_rotation()
        all_taskset = task_managers.taskset

        progress = task_managers.progress
        if not hasattr(self, '_progress'):
            self._progress = progress
        else:
            self._progress = torch.where(
                progress > self._progress,
                progress,
                self._progress,
            )

        completion = self.completion

        assignment = self.controller.memo.get('assignment', None)
        if assignment is None:
            raise ValueError("Can't find assignment")
        if len(assignment) != len(satellites):
            raise ValueError("Assignment and num satellites not matched.")

        reverse_assignment = defaultdict(list)
        for idx, elem in enumerate(assignment):
            reverse_assignment[elem].append(idx)

        sat_datas = []
        satellites = constellation.sort()

        for sat, task_id in zip(satellites, assignment):
            position_bp_n, _ = sat.rv
            mrp_attitude_bn = sat.mrp_attitude_bn
            mrp_bn_pt = torch.tensor(mrp_attitude_bn)
            dcm_bn_pt = mrp2dcm(mrp_bn_pt)

            dcm_np = earth_rotation_pn.transpose(-1, -2)
            dcm_bp = torch.einsum('...ij, ...jk-> ik', dcm_bn_pt, dcm_np)
            quaternion = dcm2q(dcm_bp)

            position = torch.einsum(
                '...ij, ...j->...i',
                dcm_np.transpose(-1, -2),
                position_bp_n,
            )

            sat_data = SatelliteData(
                id=sat.id_,
                position=position,
                orientation=quaternion,
                taskId=task_id,
            )
            sat_datas.append(sat_data)

        task_datas = []
        for idx, task in enumerate(all_taskset):
            task_data = TaskData(
                id=task.id_,
                position=task.coordinate_ecef,
                lat=task.coordinate[0],
                lon=task.coordinate[1],
                completion=completion[idx],
                satId=reverse_assignment[idx]
            )
            task_datas.append(task_data)

        log_data = Socket(
            time_step=time_stamp,
            task_datas=task_datas,
            satellites_data=sat_datas,
        )
        json_dump(
            log_data,
            self._log_dir / f'{time_stamp}.json',
            indent=4,
        )
