__all__ = [
    'LMRPEarlyStopCallback',
    'ConstellationSaveCallback',
]
import pathlib

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


@CallbackRegistry.register_()
class ConstellationSaveCallback(BaseCallback):

    def __init__(self, *args, interval: int = 100, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.interval = interval

    @property
    def should_save(self) -> bool:
        return self.runner.episode % self.interval == 0

    @property
    def work_dir(self) -> pathlib.Path:
        return self.runner.work_dir / 'constellation'

    def bind(self, *args, **kwargs):
        super().bind(*args, **kwargs)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _save(self) -> None:
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
        self.runner.memo['constellation'] = new_constellation

        save_path = self.work_dir / f'constellation_{self.runner.tag}.json'
        new_constellation.dump(str(save_path))

    def after_episode(self):
        if self.should_save:
            self._save()

    def after_run(self):
        self._save()
