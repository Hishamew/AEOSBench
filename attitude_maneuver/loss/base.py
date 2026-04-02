__all__ = [
    'LossCallback',
]
import torch

from ..callbacks import BaseCallback


class LossCallback(BaseCallback):

    def __init__(self, *args, name: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def loss(self) -> None | torch.Tensor:
        return self.runner.memo['loss'].get(self.name, None)

    @loss.setter
    def loss(self, value: torch.Tensor) -> None:
        self.runner.memo['loss'][self.name] = value

    def before_episode(self):
        self.loss = torch.tensor(
            0.0,
            device=self.runner.environment.backend,
        )
