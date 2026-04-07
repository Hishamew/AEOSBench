__all__ = [
    'MLPConfigureLoadActuator',
]
import torch
from attitude_maneuver.callbacks import BaseCallback
from attitude_maneuver.registries import CallbackRegistry

from .mlp_configure import MLPPIDConfigure


@CallbackRegistry.register_()
class MLPConfigureLoadActuator(BaseCallback):

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        if not isinstance(self.runner.model, MLPPIDConfigure):
            self.runner.logger.error(
                "MLPConfigureLoadActuator only works with MLPPIDConfigure model."
            )
            raise TypeError(
                "MLPConfigureLoadActuator only works with MLPPIDConfigure model."
            )

    def before_episode(self) -> None:
        simulator = self.runner.environment.simulator
        rw = simulator.reaction_wheels
        rw_inertia = rw.moment_of_inertia_wrt_spin.squeeze()

        hub = simulator.spacecraft.hub
        sc_mass = hub.mass
        sc_inertia = hub.moment_of_inertia_matrix_wrt_body_point
        sc_inertia = torch.diagonal(sc_inertia, dim1=-2, dim2=-1)

        params_dtype = torch.get_default_dtype()
        pid_params: torch.Tensor = self.runner.model(
            sc_inertia=sc_inertia.to(params_dtype),
            rw_inertia=rw_inertia.to(params_dtype),
            sc_mass=sc_mass.to(params_dtype),
        )
        k, ki, p, integral_limit = pid_params.unbind(-1)
        self.runner.environment.simulator.configure_pid(
            self.runner.environment.simulator_state_dict,
            k,
            ki,
            p,
            integral_limit,
            self.runner.full_grad,
        )
