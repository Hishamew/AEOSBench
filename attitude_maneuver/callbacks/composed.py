__all__ = [
    'ComposedCallback',
]

from typing import Iterable

from todd.bases.configs import Config
from todd.bases.registries import BuildPreHookMixin, Item, RegistryMeta

from ..registries import CallbackRegistry
from .base import BaseCallback


@CallbackRegistry.register_()
class ComposedCallback(BaseCallback, BuildPreHookMixin):

    def __init__(
        self,
        *args,
        callbacks: Iterable[BaseCallback],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._callbacks = callbacks

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        for callback in self._callbacks:
            callback.bind(*args, **kwargs)

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        super().build_pre_hook(config, registry, item)
        callbacks = []
        for callback_config in config.callbacks:
            callback = registry.build_or_return(callback_config)
            callbacks.append(callback)
        config.callbacks = callbacks
        return config

    def before_step(self) -> None:
        super().before_step()
        for callback in self._callbacks:
            callback.before_step()

    def after_step(self) -> None:
        super().after_step()
        for callback in self._callbacks:
            callback.after_step()

    def before_episode(self) -> None:
        super().before_episode()
        for callback in self._callbacks:
            callback.before_episode()

    def after_episode(self) -> None:
        super().after_episode()
        for callback in self._callbacks:
            callback.after_episode()

    def before_run(self) -> None:
        super().before_run()
        for callback in self._callbacks:
            callback.before_run()

    def after_run(self) -> None:
        super().after_run()
        for callback in self._callbacks:
            callback.after_run()

    def should_stop(self):
        super().should_stop()
        for callback in self._callbacks:
            if callback.should_stop():
                return True
        return False
