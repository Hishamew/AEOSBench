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


# NOTE: This evaluator is used for generating satellites with a completion rate threshold. It is not used for evaluation in the paper, so we do not report its results in the paper.
class PerCompletionRateEvaluator(BaseEvaluator):

    @property
    def max_progress_per_satellite(self) -> torch.Tensor:
        return self.controller.memo['max_progress_per_satellite']

    @max_progress_per_satellite.setter
    def max_progress_persatellite(self, value: torch.Tensor) -> None:
        self.controller.memo['max_progress_per_satellite'] = value

    @property
    def succeeded_flags_per_satellite(self) -> torch.Tensor:
        max_progress = self.max_progress_per_satellite
        durations = self.controller.task_manager.taskset.durations.unsqueeze(0)
        return max_progress >= durations

    def before_run(self):
        self.max_progress_per_satellite = torch.zeros(
            self.controller.environment.num_satellites,
            self.controller.task_manager.num_all_tasks,
            dtype=torch.int
        )

    def after_step(self):
        visible_flags: torch.Tensor = self.controller.memo['is_visible']
        visible_flags[:, ~self.controller.task_manager.ongoing_flags] = False
        self.max_progress_per_satellite = (
            self.max_progress_per_satellite + visible_flags.int()
        )

    def after_run(self) -> None:
        succeeded_flags_per_satellite = self.succeeded_flags_per_satellite
        completion_rate = (
            succeeded_flags_per_satellite.float().mean(dim=1).tolist()
        )
        self.metrics.update({
            'CR_persat': completion_rate,
        })
