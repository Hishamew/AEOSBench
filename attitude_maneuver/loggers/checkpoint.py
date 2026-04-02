__all__ = [
    'CheckpointLogger',
]
import pathlib

import torch

from ..registries import LoggerRegistry
from .base import BaseLogger


@LoggerRegistry.register_()
class CheckpointLogger(BaseLogger):

    def __init__(self, *args, interval: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self._interval = interval

    def bind(self, *args, **kwargs):
        super().bind(*args, **kwargs)
        (self.work_dir / 'checkpoint').mkdir(parents=True, exist_ok=True)

    @property
    def should_save(self) -> bool:
        return self.runner.episode % self._interval == 0

    @property
    def save_dir(self) -> pathlib.Path:
        return self.work_dir / 'checkpoint' / self.runner.memo['tag']

    def _save_checkpoint(self):
        env_state = self.runner.environment.state_dict()
        model_state_dict = self.runner.model.state_dict()

        self.save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model_state_dict, self.save_dir / 'model.pth')
        torch.save(env_state, self.save_dir / 'env.pth')

    def after_episode(self):
        if not self.should_save:
            return

        self._save_checkpoint()

    def after_run(self):
        self._save_checkpoint()
