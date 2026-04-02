import torch

from ..callbacks import BaseCallback
from ..registries import CallbackRegistry


@CallbackRegistry.register_()
class OptimizeCallback(BaseCallback):

    def __init__(
        self,
        *args,
        clip_grad_norm: float | None = None,
        **kwargs,
    ) -> None:
        self._clip_grad_norm = clip_grad_norm

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        self._optimizer = self.runner.optimizer
        if self._optimizer is None:
            self.runner.logger.error(
                'Optimizer is not set in the runner. OptimizeCallback will not work.'
            )
            raise ValueError('Optimizer is not set in the runner.')

    def after_episode(self) -> None:
        loss = self.runner.memo['loss']
        total_loss: torch.Tensor = loss['training_loss']
        self.optimizer.zero_grad()
        total_loss.backward()
        for name, param in self.runner.model.named_parameters():
            if param.requires_grad == False:
                continue

            if param.grad is None:
                self.runner.logger.warning(
                    f"Grad is None, maybe some parameters are not updated. Params Name: {name}"
                )
                continue
            if torch.isnan(param.grad).any():
                self.runner.logger.error(
                    f"NaN grad detected, deported. Params Name: {name}"
                )
                raise RuntimeError(
                    f"NaN grad detected, deported. Params Name: {name}"
                )

        if self._clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.runner.model.parameters(),
                self._clip_grad_norm,
            )
        self.optimizer.step()
        self.runner.environment.clear_grad()
