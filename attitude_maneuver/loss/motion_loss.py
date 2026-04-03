import torch

from ..registries import LossRegistry
from .base import LossCallback


@LossRegistry.register_()
class MotionLoss(LossCallback):

    def __init__(
        self,
        *args,
        threshold: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._threshold = threshold

    @property
    def with_threshold(self) -> bool:
        return self._threshold is not None

    def after_step(self) -> None:
        guidance_buffer = self.runner.environment.simulator.guidance_buffer
        angular_velocity_BR_B = guidance_buffer.angular_velocity_BR_B
        if self.with_threshold:
            angle_error = 4 * torch.atan(
                torch.norm(
                    guidance_buffer.attitude_BR,
                    dim=-1,
                    keepdim=True,
                )
            )
            threshold_mask = angle_error < self._threshold
            angular_velocity_BR_B = torch.where(
                threshold_mask,
                angular_velocity_BR_B,
                torch.zeros_like(angular_velocity_BR_B),
            )

        step_motion_loss = torch.nn.functional.mse_loss(
            angular_velocity_BR_B,
            torch.zeros_like(angular_velocity_BR_B),
            reduction='sum',
        )
        step_motion_loss = step_motion_loss / self.runner.environment.num_envs

        motion_loss = self.loss + step_motion_loss
        self.loss = motion_loss
