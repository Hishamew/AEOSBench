__all__ = [
    'GitLogger',
]
from todd.patches.py_ import run

from ..registries import LoggerRegistry
from .base import BaseLogger


@LoggerRegistry.register_()
class GitLogger(BaseLogger):

    def before_run(self) -> None:
        git_commit_hash = run('git rev-parse HEAD')
        (self.work_dir / 'git_commit_hash.txt').write_text(git_commit_hash)
