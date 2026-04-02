__all__ = [
    'RunnerHolder',
]

from todd.utils import HolderMixin

from ..runner import ControllerRunner


class RunnerHolder(HolderMixin[ControllerRunner]):

    def __init__(
        self,
        *args,
        controller: ControllerRunner | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, instance=controller, **kwargs)

    @property
    def runner(self) -> ControllerRunner:
        return self._instance
