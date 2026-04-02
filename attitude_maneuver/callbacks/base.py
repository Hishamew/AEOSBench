__all__ = [
    'BaseCallback',
]
from .runner_holder import RunnerHolder


class BaseCallback(RunnerHolder):

    def before_step(self) -> None:
        pass

    def after_step(self) -> None:
        pass

    def before_episode(self) -> None:
        pass

    def after_episode(self) -> None:
        pass

    def before_run(self) -> None:
        pass

    def after_run(self) -> None:
        pass
