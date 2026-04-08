__all__ = [
    'LMRPLoadActuator',
]
import torch
from attitude_maneuver.callbacks import BaseCallback
from attitude_maneuver.registries import CallbackRegistry

from .lmrp import LearnablePIDConfigure


@CallbackRegistry.register_()
class LMRPLoadActuator(BaseCallback):

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        if not isinstance(self.runner.model, LearnablePIDConfigure):
            self.runner.logger.error(
                "LMRPLoadActuator only works with LearnablePIDConfigure model."
            )
            raise TypeError(
                "LMRPLoadActuator only works with LearnablePIDConfigure model."
            )

    @property
    def model(self) -> LearnablePIDConfigure:
        return self.runner.model

    def before_run(self) -> None:
        env = self.runner.environment
        constellation = env.constellation
        mrp_controls = [s.mrp_control for s in constellation.sort()]
        k = torch.tensor([mrp.k for mrp in mrp_controls])
        ki = torch.tensor([mrp.ki for mrp in mrp_controls])
        p = torch.tensor([mrp.p for mrp in mrp_controls])
        if self.model.with_integral_limit:
            integral_limit = torch.tensor([
                mrp.integral_limit for mrp in mrp_controls
            ])
            params = torch.stack([k, ki, p, integral_limit], dim=-1)
        else:
            params = torch.stack([k, ki, p], dim=-1)

        with torch.no_grad():
            self.model._raw_params.copy_(params.log())

    def before_episode(self) -> None:
        params: torch.Tensor = self.model()
        k, ki, p, integral_limit = params.unbind(-1)
        self.runner.environment.simulator.configure_pid(
            self.runner.environment.simulator_state_dict,
            k,
            ki,
            p,
            integral_limit,
            self.runner.full_grad,
        )
