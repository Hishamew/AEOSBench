import torch

from constellation.data import Constellation

from ..callbacks import BaseCallback
from ..registries import CallbackRegistry
from ..taskset import LocationPointingTaskset as TaskSet


@CallbackRegistry.register_()
class ResetCallback(BaseCallback):

    def __init__(
        self,
        *args,
        sample_task_interval: int,
        sample_constellation_interval: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._task_interval = sample_task_interval
        self._constellation_interval = sample_constellation_interval

    @property
    def should_sample_constellation(self) -> bool:
        return self.runner.episode % self._constellation_interval == 0

    @property
    def should_sample_taskset(self) -> bool:
        return self.runner.episode % self._task_interval == 0

    def before_episode(self) -> None:
        if self.should_sample_constellation:
            num_envs = self.runner.environment.num_envs
            p_sampled_constellation = Constellation.sample_mrp(num_envs)
            constellation = Constellation.sample(
                list(p_sampled_constellation.values()),
                num_envs,
            )
            self.runner.environment.constellation = constellation

        if self.should_sample_taskset:
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

        if self.should_sample_constellation:
            self.runner.environment.build_simulator()
        else:
            self.runner.environment.clear_grad()

        if self.should_sample_taskset:
            self.runner.environment.setup_tracking_target()
