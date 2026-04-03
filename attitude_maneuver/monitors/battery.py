__all__ = [
    'BatteryMonitor',
]
import torch

from ..registries import MonitorRegistry
from .base import BaseMonitor


@MonitorRegistry.register_()
class BatteryMonitor(BaseMonitor):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            name='battery',
            unit='%',
            **kwargs,
        )

    def _should_after_step(self) -> None:
        state_dict = self.runner.environment.simulator_state_dict
        battery = state_dict['_battery']['stored_charge_percentage']
        self.recorder.append(battery.cpu())
