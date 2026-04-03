__all__ = [
    'LRSchedulerCallback',
]
import torch
from todd.bases.configs import Config
from todd.registries.patches import LRSchedulerRegistry

from ..callbacks import BaseCallback


class LRSchedulerCallback(BaseCallback):

    def __init__(
        self,
        *args,
        lr_scheduler_config: Config,
        interval: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._lr_scheduler_config = lr_scheduler_config
        self._interval = interval

    def bind(self, *args, **kwargs):
        super().bind(*args, **kwargs)
        self._scheduler: torch.optim.lr_scheduler.LRScheduler = LRSchedulerRegistry.build(
            self._lr_scheduler_config,
            optimizer=self.runner.optimizer,
        )

    @property
    def should_step(self) -> bool:
        return self.runner.episode % self._interval == 0

    def after_episode(self):
        if not self.should_step:
            return

        self._scheduler.step()
        self.runner.memo['log']['last_lr'] = [
            f'{lr:.3e}' for lr in self._scheduler.get_last_lr()
        ]
