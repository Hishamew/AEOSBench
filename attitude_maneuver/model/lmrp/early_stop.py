from attitude_maneuver.callbacks import BaseCallback
from attitude_maneuver.registries import CallbackRegistry

from .lmrp import LearnablePIDConfigure


@CallbackRegistry.register_()
class EarlyStopCallback(BaseCallback):

    @property
    def model(self) -> LearnablePIDConfigure:
        return self.runner.model

    def should_stop(self) -> bool:
        return self.model.done.all()
