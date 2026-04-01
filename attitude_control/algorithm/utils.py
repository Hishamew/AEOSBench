__all__ = [
    'continuous_actions_sample',
    'discrete_actions_sample',
    'InputNormalizer',
]
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from satsim.simulation.gravity import Ephemeris
from satsim.utils import dcm_to_mrp, mrp_to_rotation_matrix


def continuous_actions_sample(
    mean: torch.Tensor,
    std: torch.Tensor,
    return_epsilon: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    normal = Normal(mean, std)
    actions = normal.rsample()
    if return_epsilon:
        epsilon = (actions - mean) / (std + 1e-8)
        return actions, epsilon
    return actions


def discrete_actions_sample(
    logits: torch.Tensor,
    tau=1.0,
    hard=False,
    eps=1e-10,
):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    y = y / (tau + eps)

    y_soft = F.softmax(y, dim=-1)

    if hard:
        _, indices = y_soft.max(dim=-1)
        y_hard = F.one_hot(indices, num_classes=logits.shape[-1]).float()
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft


class InputNormalizer(nn.Module):

    def __init__(
        self,
        shape: int | list[int],
        epsilon: float = 1e-5,
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
            self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self._running_mean = new_mean
        self._running_var = new_var
        self._count = new_count

    def forward(self, batched_input: torch.Tensor):

        result = (batched_input - self.running_mean
                  ) / torch.sqrt(self.running_var + self._epsilon)

        return result


def view_angle(
    position_LP_P: torch.Tensor,
    position_BP_N: torch.Tensor,
    earth_ephemeris: Ephemeris,
):
    dcm_PN = earth_ephemeris['direction_cosine_matrix_CN'].to(position_BP_N)
    position_BP_P = torch.einsum(
        '...ij, ...j -> ...i',
        dcm_PN,
        position_BP_N,
    )

    direction_LP_P = F.normalize(position_LP_P, dim=-1)
    direction_BP_P = F.normalize(position_BP_P, dim=-1)

    cosine_angle = torch.einsum(
        '...i, ...i -> ...',
        direction_BP_P,
        direction_LP_P,
    )

    angle = torch.acos(cosine_angle)
    return angle
