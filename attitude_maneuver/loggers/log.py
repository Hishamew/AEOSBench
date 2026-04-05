__all__ = [
    'LogCallback',
]
import logging
from typing import Any

from todd.loggers import Formatter

from ..registries import LoggerRegistry
from .base import BaseLogger


@LoggerRegistry.register_()
class LogCallback(BaseLogger):

    def __init__(self, *args, file_logging: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._file_logging = file_logging

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        if not self._file_logging:
            return
        file = self.work_dir / 'train.log'
        handler = logging.FileHandler(file)
        handler.setFormatter(Formatter())
        self.logger.addHandler(handler)

    def before_episode(self) -> None:
        self.runner.memo['log'] = dict()

    def after_episode(self) -> None:
        log: dict[str, Any] = self.runner.memo.pop('log', None)

        prefix = f"Episode [{self.runner.episode+1}/{self.runner.total_episode}]\n"

        message = '\n'.join(
            f"[Source={k.replace('_', ' ').title()}]\n{v}"
            for k, v in log.items()
            if v is not None
        )

        self.logger.info(prefix + message)
