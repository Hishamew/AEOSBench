__all__ = [
    "UpdateNormalizerCallback",
]
from ..callbacks import BaseCallback
from ..model import MLPPIDConfigure
from ..registries import CallbackRegistry


@CallbackRegistry.register_()
class UpdateNormalizerCallback(BaseCallback):

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        if not isinstance(self.model, MLPPIDConfigure):
            self.runner.logger.error(
                "UpdateNormalizerCallback only works with MLPPIDConfigure model."
            )
            raise TypeError(
                "UpdateNormalizerCallback only works with MLPPIDConfigure model."
            )

    @property
    def model(self) -> MLPPIDConfigure:
        return self.runner.model

    def after_episode(self):
        if self.model.need_update:
            self.model.update_normalizer()
