__all__ = [
    'BaseLogger',
]
import pathlib

from ..callbacks import BaseCallback


class BaseLogger(BaseCallback):

    @property
    def work_dir(self) -> pathlib.Path:
        return self.runner.memo['work_dir']

    @property
    def logger(self):
        return self.runner.logger
