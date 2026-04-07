__all__ = [
    'LMRPEarlyStopCallback',
]
from attitude_maneuver.callbacks import BaseCallback
from attitude_maneuver.registries import CallbackRegistry

from constellation.data import Constellation, MRPControl, Satellite, Satellites

from .lmrp import LearnablePIDConfigure


@CallbackRegistry.register_()
class LMRPEarlyStopCallback(BaseCallback):

    @property
    def model(self) -> LearnablePIDConfigure:
        return self.runner.model

    def should_stop(self) -> bool:
        return self.model.done.all()

    def after_run(self) -> None:
        if not self.model.done.all():
            self.runner.logger.warning(
                "Process is ending, but not all MRP are done."
            )

        constellation = self.runner.environment.constellation
        params = self.runner.model()
        ks, kis, ps, integral_limits = map(
            lambda x: x.cpu().tolist(), params.unbind(-1)
        )
        satellites: Satellites = []
        for sat, k, ki, p, integral_limit in zip(
            constellation.sort(),
            ks,
            kis,
            ps,
            integral_limits,
        ):
            mrp_control = sat.mrp_control
            mrp_control = MRPControl(
                k,
                ki,
                p,
                integral_limit,
            )
            new_sat = Satellite(
                sat.id_,
                sat.inertia,
                sat.mass,
                sat.center_of_mass,
                sat.orbit.id_,
                sat.orbit,
                sat.solar_panel,
                sat.sensor,
                sat.battery,
                sat.reaction_wheels,
                mrp_control,
                sat.true_anomaly,
                sat.mrp_attitude_bn,
            )
            satellites.append(new_sat)
        new_constellation = Constellation({sat.id_: sat for sat in satellites})
        self.runner.memo['final_constellation'] = new_constellation
