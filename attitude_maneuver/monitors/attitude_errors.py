import torch

from ..registries import MonitorRegistry
from .base import BaseMonitor


@MonitorRegistry.register_()
class AttitudeErrorsMonitor(BaseMonitor):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            name='attitude_errors',
            unit='rad',
            **kwargs,
        )

    def _should_after_step(self) -> None:
        guidance_buffer = self.runner.environment.simulator.guidance_buffer
        attitude_BR = guidance_buffer.attitude_BR
        attitude_errors = torch.norm(attitude_BR, dim=-1)
        self.recoder.append(attitude_errors.cpu())
