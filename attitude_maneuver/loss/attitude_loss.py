__all__ = [
    'AttitudeLoss',
]
import torch

from ..registries import LossRegistry
from .base import LossCallback


@LossRegistry.register_()
class AttitudeLoss(LossCallback):

    def after_step(self) -> None:
        guidance_buffer = self.runner.environment.simulator.guidance_buffer
        attitude_BR = guidance_buffer.attitude_BR
        step_attitude_loss = torch.nn.functional.mse_loss(
            attitude_BR, torch.zeros_like(attitude_BR), reduction='sum'
        )
        step_attitude_loss = step_attitude_loss / self.runner.environment.num_envs

        attitude_loss = self.loss + step_attitude_loss
        self.loss = attitude_loss
