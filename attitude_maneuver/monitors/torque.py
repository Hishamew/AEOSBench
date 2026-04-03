__all__ = [
    'TorqueMonitor',
]
import torch

from ..registries import MonitorRegistry
from .base import BaseMonitor


@MonitorRegistry.register_()
class TorqueMonitor(BaseMonitor):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            name='torque',
            unit='N·m',
            **kwargs,
        )

    def _should_after_step(self) -> None:
        state_dict = self.runner.environment.simulator_state_dict
        torque: torch.Tensor = state_dict['_spacecraft']['_state_effectors'][
            '_reaction_wheels']['current_torque']
        self.recorder.append(torque.squeeze().cpu())
