__all__ = [
    'ControllerHolder',
    'EvalRunnerHolder',
]

from todd.utils import HolderMixin

from ..controller import Controller
from ..eval_runner import EvalRunner


class ControllerHolder(HolderMixin[Controller]):

    def __init__(
        self,
        *args,
        controller: Controller | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, instance=controller, **kwargs)

    @property
    def controller(self) -> Controller:
        return self._instance


class EvalRunnerHolder(HolderMixin[EvalRunner]):

    def __init__(
        self,
        *args,
        eval_runner: EvalRunner | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, instance=eval_runner, **kwargs)

    @property
    def eval_runner(self) -> EvalRunner:
        return self._instance
