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
            ephemeris = self.runner.environment.initial_ephemeris
            constellation = self.runner.environment.constellation
            self.runner.environment.taskset = TaskSet.sample(
                constellation,
                ephemeris,
            )

        if self.should_sample_constellation:
            self.runner.environment.build_simulator()
        else:
            self.runner.environment.clear_grad()

        if self.should_sample_taskset:
            self.runner.environment.setup_tracking_target()
