__all__ = [
    'CompletionRateEvaluator',
    'PerCompletionRateEvaluator',
]

import torch

from .base import BaseEvaluator


class CompletionRateEvaluator(BaseEvaluator):

    @property
    def max_progress(self) -> torch.Tensor:
        return self.controller.memo['max_progress']

    @max_progress.setter
    def max_progress(self, value: torch.Tensor) -> None:
        self.controller.memo['max_progress'] = value

    def before_run(self) -> None:
        self.max_progress = self.controller.task_manager.progress

    def after_step(self) -> None:
        self.max_progress = torch.max(
            self.max_progress,
            self.controller.task_manager.progress,
        )

    def after_run(self) -> None:
        durations = self.controller.task_manager.taskset.durations
        completion_rate = (
            self.controller.task_manager.num_succeeded_tasks
            / self.controller.task_manager.num_all_tasks
        )
        weighted_completion_rate = (
            durations[self.controller.task_manager.succeeded_flags].sum()
            / durations.sum()
        )
        partial_completion_rate = self.max_progress / durations
        weighted_partial_completion_rate = (
            self.max_progress.sum() / durations.sum()
        )

        self.metrics.update({
            'CR': completion_rate,
            'WCR': weighted_completion_rate.item(),
            'PCR': partial_completion_rate.mean().item(),
            'WPCR': weighted_partial_completion_rate.item(),
        })


class PerCompletionRateEvaluator(CompletionRateEvaluator):

    def __init__(
        self,
        *args,
        taskset_split: list[int],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._taskset_split = taskset_split

    def after_run(self) -> None:
        durations = self.controller.task_manager.taskset.durations
        durations_per_env = durations.split(self._taskset_split)
        succeeded_flags = self.controller.task_manager.succeeded_flags
        succeeded_flags_per_env = succeeded_flags.split(self._taskset_split)

        metrics = []
        for nt, durations, succeeded_flags, max_progress in zip(
            self._taskset_split,
            durations_per_env,
            succeeded_flags_per_env,
            self.max_progress.split(self._taskset_split),
        ):
            completion_rate = succeeded_flags.sum() / nt
            weighted_completion_rate = durations[succeeded_flags].sum(
            ) / durations.sum()
            partial_completion_rate = max_progress / durations
            weighted_partial_completion_rate = max_progress.sum(
            ) / durations.sum()
            metrics.append({
                'CR': completion_rate.item(),
                'WCR': weighted_completion_rate.item(),
                'PCR': partial_completion_rate.mean().item(),
                'WPCR': weighted_partial_completion_rate.item(),
            })

        self.metrics.update(metric_per_env=metrics)
