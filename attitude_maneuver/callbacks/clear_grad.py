__all__ = [
    'ClearGradCallback',
]
from ..registries import CallbackRegistry
from .base import BaseCallback


@CallbackRegistry.register_()
class ClearGradCallback(BaseCallback):

    def after_step(self) -> None:
        self.runner.environment.clear_grad()
