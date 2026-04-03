__all__ = [
    'LossCollector',
]
from todd.bases.configs import Config
from todd.bases.registries import BuildPreHookMixin, Item, RegistryMeta
from torch.utils.tensorboard import SummaryWriter

from ..registries import LossRegistry
from .base import LossCallback


@LossRegistry.register_()
class LossCollector(LossCallback, BuildPreHookMixin):

    def __init__(
        self,
        *args,
        losses: list[LossCallback],
        weights: list[float],
        **kwargs,
    ) -> None:
        super().__init__(*args, name='training_loss', **kwargs)
        self._losses = losses
        self._weights = weights

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        for loss in self._losses:
            loss.bind(*args, **kwargs)

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        losses = []
        weights = []
        for name, loss_config in config.losses.items():
            loss_config.update(name=name)
            weight = loss_config.pop('weight')
            weights.append(weight)
            loss = registry.build_or_return(loss_config)
            losses.append(loss)
        config.losses = losses
        config.weights = weights
        return config

    def before_episode(self) -> None:
        super().before_episode()
        for loss in self._losses:
            loss.before_episode()

    def after_step(self) -> None:
        for loss in self._losses:
            loss.after_step()

    def after_episode(self) -> None:
        total_loss = 0
        for loss_callback, weight in zip(self._losses, self._weights):
            total_loss += weight * loss_callback.loss / self.runner.episode_length
        self.loss = total_loss

        if 'log' in self.runner.memo:
            self.runner.memo['loss']['total_loss'] = sum([
                l.loss for l in self._losses
            ])
            self.runner.memo['log']['loss'] = '\n'.join([
                f"{l.name.replace('_', ' ').title()}: {l.loss.item():.6e}"
                for l in self._losses
            ])

        if 'tensorboard' in self.runner.memo:
            writer: SummaryWriter = self.runner.memo['tensorboard']
            for loss_callback in self._losses:
                writer.add_scalar(
                    f"Train/{loss_callback.name.replace('_', ' ').title()}",
                    loss_callback.loss.item(),
                    self.runner.episode,
                )
