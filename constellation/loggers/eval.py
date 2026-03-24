__all__ = [
    'EvalLogger',
]
import os

import todd
from todd.patches.py_ import json_dump

from .base import BaseLogger


class EvalLogger(BaseLogger):

    def after_run(self) -> None:
        id_ = self.controller.memo['current_id']
        metrics = self.controller.memo['metrics']

        if os.environ['RANK'] == '0':
            todd.logger.info(
                f"rank %s {id_=}\n{metrics=}",
                os.environ['RANK'],
            )
        json_path = self._work_dir / f'{id_ // 1000:02d}' / f'{id_:05d}.json'
        json_dump(metrics, str(json_path))
