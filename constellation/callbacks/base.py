__all__ = [
    'BaseCallback',
    'EvalRunnerCallback',
]

from .controller_holder import ControllerHolder, EvalRunnerHolder


class BaseCallback(ControllerHolder):

    def should_break(self) -> bool:
        return False

    def before_step(self) -> None:
        pass

    def after_step(self) -> None:
        pass

    def before_run(self) -> None:
        pass

    def after_run(self) -> None:
        pass


class EvalRunnerCallback(EvalRunnerHolder):

    def should_break(self) -> bool:
        return False

    def before_step(self) -> None:
        pass

    def after_step(self) -> None:
        pass

    def before_run(self) -> None:
        pass

    def after_run(self) -> None:
        pass
