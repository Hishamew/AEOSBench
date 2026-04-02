__all__ = [
    'TensorboardLogger',
]
from torch.utils.tensorboard import SummaryWriter

from ..registries import LoggerRegistry
from .base import BaseLogger


@LoggerRegistry.register_()
class TensorboardLogger(BaseLogger):

    def before_run(self):
        self._writer = SummaryWriter(log_dir=self.work_dir / 'tensorboard')
        self.runner.memo['tensorboard'] = self._writer

    def after_run(self):
        self._writer.close()
