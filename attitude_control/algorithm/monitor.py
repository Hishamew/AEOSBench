import os

import matplotlib.pyplot as plt
import numpy
import torch

from satsim.architecture import constants

from .base import BaseStateMonitor
from .constellation.constellation import \
    SpacecraftPointingConstellationStateDict
from .environment.base_environment import Observation
from .environment.environment import SpacecraftPointingEnvironment
from .registry import MonitorRegistry


@MonitorRegistry.register_()
class BatteryMonitor(BaseStateMonitor):

    def __init__(self) -> None:
        super().__init__(
            'Battery Percentage',
            '%',
        )

    def pick_state(self, environment, observation, state_dict):
        return observation['battery_percentage']


@MonitorRegistry.register_()
class AttitudeErrorMonitor(BaseStateMonitor):

    def __init__(self) -> None:
        super().__init__(
            'Angle Error',
            'rad',
        )

    def _post_hook(self, state: torch.Tensor) -> torch.Tensor:
        state = 4 * torch.atan(state.norm(dim=-1))
        return super()._post_hook(state)

    def pick_state(self, environment, observation, state_dict):
        return environment.calculate_real_attitude_error()


@MonitorRegistry.register_()
class TorqueMonitor(BaseStateMonitor):

    def __init__(self) -> None:
        super().__init__(
            'Torque',
            'N*m',
        )

    def pick_state(self, environment, observation, state_dict) -> torch.Tensor:
        return state_dict['_spacecraft']['_state_effectors'][
            '_reaction_wheels']['current_torque']

    def _post_hook(self, state: torch.Tensor) -> torch.Tensor:
        return super()._post_hook(state.squeeze())


@MonitorRegistry.register_()
class AltitudeMonitor(BaseStateMonitor):

    def __init__(self) -> None:
        super().__init__(
            'Altitude',
            'm',
        )

    def pick_state(self, environment, observation, state_dict):
        return state_dict['_spacecraft']['_hub']['dynamic_params'][
            'position_BP_N']

    def _post_hook(self, state: torch.Tensor) -> torch.Tensor:
        state = state.norm(dim=-1) - constants.REQ_EARTH * 1e3
        return super()._post_hook(state)


@MonitorRegistry.register_()
class RelativePositionMonitor(BaseStateMonitor):

    def __init__(self) -> None:
        super().__init__(
            'relative position',
            'm',
        )

    def pick_state(
        self,
        environment: SpacecraftPointingEnvironment,
        observation: Observation,
        state_dict: SpacecraftPointingConstellationStateDict,
    ):
        position_CN_N = state_dict['_chief_spacecraft']['_hub'][
            'dynamic_params']['position_BP_N']
        position_DN_N = state_dict['_spacecraft']['_hub']['dynamic_params'][
            'position_BP_N']

        return position_CN_N - position_DN_N

    def plot(
        self,
        save_path,
        start_time=None,
        end_time=None,
    ):
        relative_trace = numpy.array(self._state_list)
        fig = plt.figure(figsize=(15, 15))
        for i in range(self._num):
            trace = numpy.array(relative_trace[i])
            ax = fig.add_subplot(3, 3, i + 1, projection='3d')
            x, y, z = (
                trace[..., 0],
                trace[..., 1],
                trace[..., 2],
            )
            ax.plot(
                x,
                y,
                z,
                color='#2E86AB',
                linewidth=2,
                label='relative_trace',
            )
            ax.scatter(
                x,
                y,
                z,
                color='gray',
                s=30,
                alpha=0.7,
            )
            ax.scatter(x[0],
                       y[0],
                       z[0],
                       color='green',
                       s=100,
                       label='start',
                       edgecolors='black')
            ax.scatter(
                x[-1],
                y[-1],
                z[-1],
                color='red',
                s=100,
                label='end',
                edgecolors='black',
            )
            for i in range(0, len(trace) - 1, 30):  # 每隔2个点画一个箭头，可调整步长
                ax.quiver(
                    x[i],
                    y[i],
                    z[i],
                    x[i + 1] - x[i],
                    y[i + 1] - y[i],
                    z[i + 1] - z[i],
                    color='purple',
                    length=1,
                    normalize=True,
                    arrow_length_ratio=0.15,
                    alpha=0.6,
                )

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Env {i+1} {self._state_name} ({self._state_unit})')

            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{self._state_name}.png'))
        plt.close()

        self._state_list = [[] for i in range(self._num)]
