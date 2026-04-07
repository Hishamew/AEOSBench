from ..callbacks import BaseCallback
from ..model import MLPPIDConfigure
from ..registries import CallbackRegistry


@CallbackRegistry.register_()
class UpdateNormalizerCallback(BaseCallback):

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        self._model = self.runner.model
        if not isinstance(self._model, MLPPIDConfigure):
            self.runner.logger.error(
                "UpdateNormalizerCallback only works with MLPPIDConfigure model."
            )
            raise TypeError(
                "UpdateNormalizerCallback only works with MLPPIDConfigure model."
            )

    def after_episode(self):
        if self._model.need_update:
            self._model.update_normalizer()
