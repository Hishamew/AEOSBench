__all__ = [
    'BatteryLoss',
]
import torch

from ..registries import LossRegistry
from .base import LossCallback


@LossRegistry.register_()
class BatteryLoss(LossCallback):

    def after_step(self) -> None:
        state_dict = self.runner.environment.simulator_state_dict
        true_torque = state_dict['_spacecraft']['_state_effectors'][
            '_reaction_wheels']['current_torque']
        step_battery_loss = torch.sum(torch.abs(true_torque))

        battery_loss = self.loss
        battery_loss += step_battery_loss / self.runner.environment.num_envs
        self.loss = battery_loss
