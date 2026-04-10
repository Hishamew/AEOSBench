__all__ = [
    'ParallelSatsimEnvironment',
]
import torch

from ...data import (
    Constellation,
    TaskSet,
    concat_constellations,
    concat_tasksets,
)
from .environment import SatsimEnvironment


class ParallelSatsimEnvironment(SatsimEnvironment):

    def __init__(
        self,
        *args,
        constellations: list[Constellation],
        tasksets: list[TaskSet],
        **kwargs,
    ) -> None:
        parallel_constellation = concat_constellations(constellations)
        parallel_taskset = concat_tasksets(tasksets)
        super().__init__(
            *args,
            constellation=parallel_constellation,
            all_tasks=parallel_taskset,
            **kwargs
        )
        self._origin_constellations = constellations
        self._origin_tasksets = tasksets

    @property
    def mask(self) -> torch.BoolTensor:
        num_sats = [
            len(constellation) for constellation in self._origin_constellations
        ]
        num_tasks = [len(taskset) for taskset in self._origin_tasksets]
        return torch.block_diag(
            *[
                torch.ones(m, n, dtype=torch.bool, device=self._backend)
                for m, n in zip(num_sats, num_tasks)
            ]
        )

    def is_visible(self, tasks: TaskSet) -> torch.Tensor:
        visible = super().is_visible(tasks)
        return visible & self.mask
