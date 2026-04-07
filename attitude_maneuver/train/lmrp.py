import pathlib

import torch
from todd.patches.py_ import json_load

from constellation.constants import (
    BENCHMARK_ANNOTATIONS_ROOT,
    BENCHMARK_CONSTELLATIONS_ROOT,
)
from constellation.data.constellations import Constellation

from ..callbacks import BaseCallback
from ..registries import CallbackRegistry
from ..taskset import LocationPointingTaskset as TaskSet


@CallbackRegistry.register_()
class LMRPResetCallback(BaseCallback):

    def __init__(
        self,
        *args,
        reset_interval: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reset_interval = reset_interval

    @property
    def should_reset(self) -> bool:
        return self.runner.episode % self._reset_interval == 0

    def before_episode(self) -> None:
        ephemeris = self.runner.environment.ephemeris
        if self.runner.environment.is_built:
            state_dict = self.runner.environment.simulator_state_dict
            position_BP_N = state_dict['_spacecraft']['_hub'][
                'dynamic_params']['position_BP_N']
        else:
            constellation = self.runner.environment.constellation
            rs = []
            for sat in constellation.sort():
                r, _ = sat.rv
                r = torch.from_numpy(r)
                rs.append(r)
            position_BP_N = torch.stack(rs)

        self.runner.environment.taskset = TaskSet.sample(
            position_BP_N,
            ephemeris,
        )
        if self.should_reset:
            self.runner.environment.build_simulator()
        else:
            self.runner.environment.clear_grad()

        self.runner.environment.setup_tracking_target()


# TODO: This is for lmrp test
@CallbackRegistry.register_()
class ConstellationLoader(BaseCallback):

    def before_run(self) -> None:
        annotations: list[int] = json_load(
            str(BENCHMARK_ANNOTATIONS_ROOT / f'val_seen.json'),
        )['ids']
        id_ = annotations[0]
        constellation_path = (
            BENCHMARK_CONSTELLATIONS_ROOT / 'val_seen' / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        constellation = Constellation.load(str(constellation_path))

        self.runner.environment.constellation = constellation
