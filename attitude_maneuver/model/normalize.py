import torch
from torch import nn


class InputNormalizer(nn.Module):

    def __init__(
        self,
        shape: int | list[int],
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        shape = [shape] if isinstance(shape, int) else shape
        self.register_buffer('_running_mean', torch.zeros(*shape))
        self.register_buffer('_running_var', torch.ones(*shape))
        self.register_buffer('_count', torch.tensor(epsilon))
        self._epsilon = epsilon

    @property
    def running_mean(self) -> torch.Tensor:
        return self.get_buffer('_running_mean')

    @property
    def running_var(self) -> torch.Tensor:
        return self.get_buffer('_running_var')

    @property
    def count(self) -> torch.Tensor:
        return self.get_buffer('_count')

    @torch.no_grad()
    def update(self, batched_input: torch.Tensor) -> None:
        if batched_input.dim() < 2:
            batched_input = batched_input.unsqueeze(0)

        batch_mean = torch.mean(batched_input, dim=0)
        batch_var = torch.var(batched_input, dim=0, unbiased=False)
        batch_count = batched_input.size(0)

        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count

        new_mean = self.running_mean + delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta * delta * self.count * batch_count / (
            self.count + batch_count
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self._running_mean = new_mean
        self._running_var = new_var
        self._count = new_count

    def forward(self, batched_input: torch.Tensor):

        result = (batched_input - self.running_mean
                  ) / torch.sqrt(self.running_var + self._epsilon)

        return result
