__all__ = [
    'ModelAlgorithm',
    'Batch',
]

from typing import Any, Mapping, NamedTuple

import einops
import todd
import torch
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

from ..constants import STATISTICS_PATH
from ..new_transformers import Model, Statistics
from .base import Observation, VecAlgorithm


class Batch(NamedTuple):
    time_step: torch.Tensor
    constellation_sensor_type: torch.Tensor
    constellation_sensor_enabled: torch.Tensor
    constellation_data: torch.Tensor
    constellation_mask: torch.Tensor
    tasks_sensor_type: torch.Tensor
    tasks_data: torch.Tensor
    tasks_mask: torch.Tensor


class ModelAlgorithm(VecAlgorithm):

    def __init__(
        self,
        *args,
        greedy: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.actor = Model()

        self._statistics: Statistics = torch.load(
            STATISTICS_PATH,
            weights_only=False,
        )
        self._greedy = greedy

        if todd.Store.cuda:
            self.cuda()

    def _extract_features(
        self,
        observations: list[Observation],
    ) -> Batch:
        constellation_sensor_type = pad_sequence(
            [obs['constellation_sensor_type'] for obs in observations],
            batch_first=True,
        )
        constellation_sensor_enabled = pad_sequence(
            [obs['constellation_sensor_enabled'] for obs in observations],
            batch_first=True,
        )
        constellation_data = pad_sequence(
            [obs['constellation_data'] for obs in observations],
            batch_first=True,
        )
        constellation_data = (
            constellation_data - self._statistics.constellation_mean
        ) / (self._statistics.constellation_std + 1e-6)

        num_constellations = torch.tensor([
            obs['num_satellites'] for obs in observations
        ])
        max_len = constellation_data.size(1)
        mask = torch.arange(max_len).expand(
            num_constellations.size(0), max_len
        ) < einops.rearrange(num_constellations, 'ne -> ne 1')

        task_sensor_type = pad_sequence(
            [obs['tasks_sensor_type'] for obs in observations],
            batch_first=True,
        )
        tasks_data = pad_sequence(
            [obs['tasks_data'] for obs in observations],
            batch_first=True,
        )
        tasks_data = (tasks_data - self._statistics.taskset_mean
                      ) / (self._statistics.taskset_std + 1e-6)

        num_tasks = torch.tensor([obs['num_tasks'] for obs in observations])
        task_mask = torch.arange(tasks_data.size(1)).expand(
            num_tasks.size(0), tasks_data.size(1)
        ) < einops.rearrange(num_tasks, 'nt -> nt 1')

        return Batch(
            time_step=torch.tensor(obs['time_step'] for obs in observations),
            constellation_sensor_type=constellation_sensor_type,
            constellation_sensor_enabled=constellation_sensor_enabled.int(),
            constellation_data=constellation_data,
            constellation_mask=mask,
            tasks_sensor_type=task_sensor_type,
            tasks_data=tasks_data,
            tasks_mask=task_mask,
        )

    def step(
        self,
        observations: list[Observation],
    ) -> torch.Tensor:
        # Extract features from taskset and constellation
        batch = self._extract_features(observations)
        if todd.Store.cuda:
            batch = Batch(
                *[tensor.clone().detach().cuda() for tensor in batch]
            )
        logits = self.actor.predict(*batch)
        distributions = Categorical(logits=logits)

        probs: torch.Tensor = distributions.probs
        actions = probs.argmax(-1)

        return actions - 1

    def load(self, state_dict: Mapping[str, Any]):
        self.actor.load_state_dict(state_dict)
